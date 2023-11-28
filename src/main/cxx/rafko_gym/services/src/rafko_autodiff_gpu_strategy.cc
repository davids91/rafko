/*! This file is part of davids91/Rafko.
 *
 *    Rafko is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Rafko is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */
#include "rafko_gym/services/rafko_autodiff_gpu_strategy.hpp"

#include <atomic>
#include <cmath>
#include <memory>
#include <set>

#include "rafko_backpropagation_operation.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_utilities/models/rafko_gpu_kernel_library.hpp"
#include "rafko_utilities/services/rafko_string_utils.hpp"
#include "rafko_utilities/services/thread_group.hpp"
#include "spike_function.hpp"
#include "transfer_function.hpp"

namespace {
template <typename T>
std::size_t doubles_needed_for_vector(const std::vector<T> &vector) {
  return (vector.size() + 1u) / (sizeof(double) / sizeof(T));
  /*!Note: in the (size + 1)/sizeof(double) ratio, the +1 makes sure that there
   * are enough elements in the array, and the size is not smaller due to the
   * integer division */
}

std::size_t get_safe_global_work_dimension(const cl::Device &device) {
  auto dimensions = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  return std::min({dimensions[0], dimensions[1], dimensions[2]});
}

} // namespace

namespace rafko_gym {

AutoDiffGPUStrategy::AutoDiffGPUStrategy(
    const cl::Device &device, const rafko_mainframe::RafkoSettings &settings,
    rafko_net::RafkoNet &network,
    const std::vector<std::uint32_t> &neuron_index_to_spike_op_map,
    std::shared_ptr<RafkoDataSet> data_set)
    : m_settings(settings), m_network(network),
      m_neuronIndexToSpikeOperationIndex(neuron_index_to_spike_op_map),
      m_maxWorkItemSize(get_safe_global_work_dimension(device)),
      m_maxAllocatableBytes(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) {
  if (data_set)
    set_data_set(data_set);
}

std::size_t AutoDiffGPUStrategy::get_d_w_threads_count() const {
  const std::size_t input_buffer_byte_size =
      /* Inputs */
      sizeof(double) *
          ((m_dataSet->get_number_of_sequences() *
            m_dataSet->get_inputs_in_one_sequence() *
            m_network.input_data_size()) +
           /* Labels */
           sizeof(double) * (m_dataSet->get_number_of_sequences() *
                             m_dataSet->get_sequence_size() *
                             m_network.output_neuron_number()) +
           /* Neuron index to Spike operation map + propagation instructions */
           sizeof(double) *
               doubles_needed_for_vector(m_neuronIndexToSpikeOperationIndex) +
           sizeof(double) *
               doubles_needed_for_vector(m_neuralPropagationInstructions)) +
      /* Metadata */
      sizeof(double) * 2;
  const std::size_t operation_arrays_byte_size =
      sizeof(double) *
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations);
  const std::size_t weight_derivatives_byte_size =
      sizeof(double) *
      static_cast<std::uint32_t>(m_network.weight_table_size());
  const std::size_t available_bytes =
      (m_maxAllocatableBytes -
       (input_buffer_byte_size + operation_arrays_byte_size +
        weight_derivatives_byte_size));
  return std::min({static_cast<std::size_t>(m_network.weight_table_size()),
                   static_cast<std::size_t>(m_maxWorkItemSize),
                   (available_bytes / operation_arrays_byte_size)});
}

std::vector<rafko_mainframe::RafkoNBufShape>
AutoDiffGPUStrategy::get_input_shapes() const {
  RFASSERT(static_cast<bool>(m_dataSet));
  RFASSERT_LOG(
      "Autodiff strategy Input shape: (weights: {} + inputs: {} + Labels: {} + "
      "sequence start: + sequence truncation: {} + Neural propagation "
      "instructions: {} + Neuron index to Spike operation index "
      "mapping: {})",
      /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
      /* Inputs */
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
      /* Labels */
      (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_network.output_neuron_number()),
      /* Neuron index to Spike operation map + propagation instructions */
      doubles_needed_for_vector(m_neuronIndexToSpikeOperationIndex),
      doubles_needed_for_vector(m_neuralPropagationInstructions),
      /*!Note: both vectors have a base type different, than double! Because of
       *the (possible) size difference Indexing has to be checked carefully, and
       *documented explicitly to help checking by-byte alignment!
       **/
      /* Metadata */
      /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u);
  return {rafko_mainframe::RafkoNBufShape{
      /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
      /* Inputs */
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
      /* Labels */
      (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_network.output_neuron_number()),
      /* Neuron index to Spike operation map + propagation instructions */
      doubles_needed_for_vector(m_neuronIndexToSpikeOperationIndex),
      doubles_needed_for_vector(m_neuralPropagationInstructions),
      /* Metadata */
      /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u}};
}

std::vector<rafko_mainframe::RafkoNBufShape>
AutoDiffGPUStrategy::get_output_shapes() const {
  const std::size_t d_w_threads = get_d_w_threads_count();
  RFASSERT_LOG("Autdiff GPU Strategy output buffer overall size: (op values: "
               "{} + op derivatives: ({} * {}) + w derivatives: {})",
               /* operation values */
               (m_dataSet->get_number_of_sequences() *
                m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations),
               /* operation derivatives */
               (m_dataSet->get_number_of_sequences() *
                m_dataSet->get_sequence_size() * m_numberOfOperations),
               d_w_threads,
               /* Weight derivatives */
               static_cast<std::uint32_t>(m_network.weight_table_size()));
  return {rafko_mainframe::RafkoNBufShape{
      /* operation values */
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations),
      /* operation derivatives */
      (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_numberOfOperations * d_w_threads),
      /* Weight derivatives */
      static_cast<std::uint32_t>(m_network.weight_table_size())}};
}

std::tuple<cl::NDRange, cl::NDRange, cl::NDRange>
AutoDiffGPUStrategy::get_solution_space() const {
  RFASSERT(m_built);
  RFASSERT(static_cast<bool>(m_dataSet));
  const std::size_t work_items[2] = {
      (std::min(m_maxWorkItemSize,
                (std::min(m_settings.get_minibatch_size(),
                          m_dataSet->get_number_of_sequences()) *
                 m_maximumLocalWorkers))),
      get_d_w_threads_count()};
  RFASSERT_LOG(
      "Autodiff strategy global solution space: "
      "NDRange(min({}, (min({}, {}) * {})) = {}, min(weight_count = {}, {}))",
      m_maxWorkItemSize, m_settings.get_minibatch_size(),
      m_dataSet->get_number_of_sequences(), m_maximumLocalWorkers,
      work_items[0], m_network.weight_table_size(), m_maxWorkItemSize);
  return {cl::NullRange /*offset*/,
          cl::NDRange(work_items[0], work_items[1]) /*global*/,
          cl::NDRange(m_maximumLocalWorkers, 1) /*local*/};
}

std::vector<std::vector<std::uint32_t>>
AutoDiffGPUStrategy::generate_operation_paralell_matrix(
    const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
        &operations) {
  using DepthMarker = std::unique_ptr<std::atomic<std::uint32_t>>;

  const std::uint32_t number_of_threads =
      m_settings.get_max_solve_threads() *
      m_settings.get_max_processing_threads(); /* Educated guess based on the
                                                  average available CPU cores */
  rafko_utilities::ThreadGroup process_threads(number_of_threads);
  std::vector<DepthMarker> operations_depth;

  operations_depth.reserve(operations.size());
  for (std::uint32_t operation_index = 0u; operation_index < operations.size();
       ++operation_index) {
    operations_depth.emplace_back(
        std::make_unique<std::atomic<std::uint32_t>>());
  }

  std::atomic<std::uint32_t> modified_operations{
      1}; /* not 0, to enter into the loop */
  std::atomic<std::uint32_t> max_depth{0};
  while (0 < modified_operations) {
    modified_operations = 0;
    process_threads.start_and_block(
        [number_of_threads, &operations, &operations_depth, &max_depth,
         &modified_operations](std::uint32_t thread_index) {
          const std::uint32_t operations_in_one_group =
              1 + (operations.size() / number_of_threads);
          const std::uint32_t operations_start_index =
              operations_in_one_group * thread_index;
          const std::uint32_t operations_in_this_group = std::min(
              operations_in_one_group,
              static_cast<std::uint32_t>(std::max(
                  0, static_cast<std::int32_t>(operations.size()) -
                         static_cast<std::int32_t>(operations_start_index))));
          for (std::uint32_t operation_index = operations_start_index;
               operation_index <
               (operations_start_index + operations_in_this_group);
               ++operation_index) {
            for (RafkoBackpropagationOperation::Dependency &d :
                 operations[operation_index]->get_dependencies()) {
              if (*operations_depth[operation_index] <=
                  *operations_depth[d->get_operation_index()]) {
                operations_depth[operation_index]->store(
                    *operations_depth[d->get_operation_index()] + 1u);
                modified_operations.fetch_add(1u);
                max_depth.store(
                    std::max(max_depth, *operations_depth[operation_index]));
              }
            } /*for(every dependency in that operation)*/
          }   /*for(every operation)*/
        });
  } /*while(there are modified operations)*/

  std::vector<std::vector<std::uint32_t>> operations_matrix(max_depth + 1);
  for (std::uint32_t operation_index = 0u; operation_index < operations.size();
       ++operation_index) {
    operations_matrix[*operations_depth[operation_index]].push_back(
        operation_index);
  }

  RFASSERT_LOGV2(operations_matrix, "Operations matrix:");
  return operations_matrix;
}

std::vector<std::uint32_t>
AutoDiffGPUStrategy::generate_propagation_instructions(
    const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
        &operations) {
  std::vector<std::vector<std::uint32_t>> operations_matrix =
      generate_operation_paralell_matrix(operations);
  std::vector<std::uint32_t> result;

  result.reserve(operations_matrix.size() * (operations.size() + 1u));
  for (const std::vector<std::uint32_t> &operation_row : operations_matrix) {
    result.push_back(operation_row.size());

    for (std::uint32_t operation_index : operation_row) {
      const std::shared_ptr<RafkoBackpropagationOperation> &operation =
          operations[operation_index];
      switch (operation->get_type()) {
      case ad_operation_neuron_bias_d:
        result.insert(
            result.end(),
            {ad_operation_neuron_bias_d,
             std::static_pointer_cast<RafkoBackpropNeuronBiasOperation>(
                 operation)
                 ->get_weight_index(),
             0u,
             std::static_pointer_cast<RafkoBackpropNeuronBiasOperation>(
                 operation)
                 ->get_dependency_descriptor(),
             operation->get_operation_index(),
             static_cast<std::uint32_t>(
                 std::static_pointer_cast<RafkoBackpropNeuronBiasOperation>(
                     operation)
                     ->get_input_function())});
        break;
      case ad_operation_neuron_input_d: {
        auto upcasted_operation_ptr =
            std::static_pointer_cast<RafkoBackpropNeuronInputOperation>(
                operation);
        RFASSERT(1u <=
                 upcasted_operation_ptr->get_own_dependencies_past_included()
                     .size());
        result.insert(
            result.end(),
            {ad_operation_neuron_input_d, upcasted_operation_ptr->m_weightIndex,
             upcasted_operation_ptr->get_f_x_dependency_index(),
             upcasted_operation_ptr->get_u_x_dependency_index(),
             operation->get_operation_index(),
             ((upcasted_operation_ptr->get_input_function() & 0x00FFu) |
              ((upcasted_operation_ptr->get_input_past_index() << 8) &
               0xFF00u))});
      } break;
      case ad_operation_objective_d: {
        RFASSERT(1u == operation->get_own_dependencies().size());
        auto upcasted_operation_ptr =
            std::static_pointer_cast<RafkoBackpropObjectiveOperation>(
                operation);
        result.insert(
            result.end(),
            {ad_operation_objective_d,
             upcasted_operation_ptr->get_label_index(),
             upcasted_operation_ptr->get_sample_number(),
             operation->get_own_dependencies()[0]->get_operation_index(),
             operation->get_operation_index(),
             static_cast<std::uint32_t>(
                 upcasted_operation_ptr->get_cost_type())});
      } break;
      case ad_operation_neuron_spike_d:
        RFASSERT(1u == operation->get_own_dependencies().size());
        result.insert(
            result.end(),
            {ad_operation_neuron_spike_d,
             std::static_pointer_cast<RafkoBackpropSpikeFnOperation>(operation)
                 ->get_weight_index(),
             0u, operation->get_own_dependencies()[0]->get_operation_index(),
             operation->get_operation_index(),
             static_cast<std::uint32_t>(
                 std::static_pointer_cast<RafkoBackpropSpikeFnOperation>(
                     operation)
                     ->get_spike_function())});
        break;
      case ad_operation_neuron_transfer_d:
        RFASSERT(1u == operation->get_own_dependencies().size());
        result.insert(
            result.end(),
            {ad_operation_neuron_transfer_d, 0u, 0u,
             operation->get_own_dependencies()[0]->get_operation_index(),
             operation->get_operation_index(),
             static_cast<std::uint32_t>(
                 std::static_pointer_cast<RafkoBackpropTransferFnOperation>(
                     operation)
                     ->get_transfer_function())});
        break;
      case ad_operation_network_weight_regularization_feature:
        break; // TODO: Features
      case ad_operation_network_feature:
        break;
      default:
        break;
      }
    } /* for(each operation in the row) */
  }   /* for(each row in the operation) */
  return result;
}

std::string AutoDiffGPUStrategy::generate_value_kernels(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size,
    const rafko_mainframe::RafkoSettings &settings) {
  std::string kernel_source = "switch(network_instructions[0]){";
  kernel_source +=
      "case ad_operation_neuron_bias_d:{" +
      RafkoBackpropNeuronBiasOperation::generic_value_kernel_operation(
          weight_array, operations_value_array,
          "==behavior_data==" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_input_d:{" +
      RafkoBackpropNeuronInputOperation::generic_value_kernel_operation(
          network_input_array, weight_array, operations_value_array,
          operations_array_size,
          "(==behavior_data== & 0x00FFu)" /*behavior_index*/
          ) +
      "}break;";

  /* RafkoBackpropObjectiveOperation does not calculate value */

  kernel_source +=
      "case ad_operation_neuron_spike_d:{" +
      RafkoBackpropSpikeFnOperation::generic_value_kernel_operation(
          weight_array, operations_value_array, operations_array_size,
          "==behavior_data==" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_transfer_d:{" +
      RafkoBackpropTransferFnOperation::generic_value_kernel_operation(
          operations_value_array, "==behavior_data==" /*behavior_index*/,
          settings) +
      "}break;";
  kernel_source += "}";

  substitute_index_values_in_kernels(kernel_source);

  // TODO: Weight regularization + solution feature operation
  return kernel_source;
}

std::string AutoDiffGPUStrategy::generate_derivative_kernels(
    std::string network_input_array, std::string label_array,
    std::string weight_array, std::string operations_value_array,
    std::string operations_derivative_array, std::string operations_array_size,
    const rafko_mainframe::RafkoSettings &settings) {
  std::string kernel_source = "switch(network_instructions[0]){";
  kernel_source +=
      "case ad_operation_neuron_bias_d:{" +
      RafkoBackpropNeuronBiasOperation::generic_derivative_kernel_operation(
          weight_array, operations_value_array, operations_derivative_array,
          "==behavior_data==" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_input_d:{" +
      RafkoBackpropNeuronInputOperation::generic_derivative_kernel_operation(
          network_input_array, weight_array, operations_value_array,
          operations_derivative_array, operations_array_size,
          "==behavior_data==" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_objective_d:{" +
      RafkoBackpropObjectiveOperation::generic_derivative_kernel_operation(
          label_array, operations_value_array, operations_derivative_array,
          "==behavior_data==" /*behavior_index*/,
          "==number_of_sequences==" /*sample_number*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_spike_d:{" +
      RafkoBackpropSpikeFnOperation::generic_derivative_kernel_operation(
          weight_array, operations_value_array, operations_derivative_array,
          operations_array_size, "==behavior_data==" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_transfer_d:{" +
      RafkoBackpropTransferFnOperation::generic_derivative_kernel_operation(
          operations_value_array, operations_derivative_array,
          "==behavior_data==" /*behavior_index*/, settings) +
      "}break;";
  kernel_source += "}";

  substitute_index_values_in_kernels(kernel_source);

  // TODO: Weight regularization + feature operation
  return kernel_source;
}

void AutoDiffGPUStrategy::substitute_index_values_in_kernels(
    std::string &kernel_source) {
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_input_index=="),
      "network_instructions[2]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==this_op_weight_index=="),
      "network_instructions[1]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==op_index=="), "network_instructions[4]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==past_index=="),
      "((==behavior_data== & 0xFF00u) >> 8)");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==f_x_op_index=="), "network_instructions[2]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==u_x_op_index=="), "network_instructions[3]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==label_index=="), "network_instructions[1]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==dependency_op_index=="),
      "network_instructions[3]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==dependency_descriptor=="),
      "network_instructions[3]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==behavior_data=="),
      "network_instructions[5]");
}

void AutoDiffGPUStrategy::build(const std::vector<OperationsType> &operations,
                                std::uint32_t weight_relevant_operation_count) {
  RFASSERT(
      m_network.neuron_array_size() ==
      static_cast<std::int32_t>(m_neuronIndexToSpikeOperationIndex.size()));
  std::string kernel_source =
      rafko_utilities::atomic_double_add_function +
      rafko_utilities::atomic_double_average_function +
      rafko_net::InputFunction::get_kernel_enums() +
      rafko_net::TransferFunction::get_kernel_enums() +
      rafko_net::SpikeFunction::get_kernel_enums() +
      CostFunction::get_kernel_enums() +
      RafkoBackpropagationOperation::get_kernel_enums() +
      rafko_utilities::random_function + R"(

    __global double triggered_derivative_operations;
    __constant bool evaluate_network = true;
    /*!Note: The only purpose of the autodiff operations is to train the network
     * so evaluation is always true in this context; "Simple" network inference
     * is not strictly on topic.
     */
    void execute_derivative_workers(
      int d_w_index, int available_memory_slots, int weight_table_size, bool save_to_output,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array,
      __constant unsigned int* network_instruction_table, __constant unsigned int* neuron_index_to_spike_op_map,
      __local double* local_derivative_sum
    ){
      ==operation_locals==
      const int operation_count = ==operation_count==;
      const unsigned int network_instruction_count = ==network_instruction_count==;
      int current_instruction_index = 0;

      //reset current operation derivatives
      int operation_index = get_local_id(0);
      while(operation_index < operation_count){
        operations_d_array[operation_index] = 0.0;
        operation_index += get_local_size(0);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);

      while(current_instruction_index < network_instruction_count){
        int operations_to_do_now = network_instruction_table[current_instruction_index];
        int local_operation_index = 0;
        while((local_operation_index + get_local_id(0)) < operations_to_do_now){
          int current_instruction_entry_start = (
            current_instruction_index + 1
            + ((local_operation_index + get_local_id(0)) * ==one_neural_instruction_entry_size==)
          );
          __constant unsigned int* network_instructions = &network_instruction_table[current_instruction_entry_start];

          ==derivative_command_list_parsers== 

          local_operation_index += get_local_size(0);
        }
        current_instruction_index += 1 + (operations_to_do_now * ==one_neural_instruction_entry_size==);
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      barrier(CLK_GLOBAL_MEM_FENCE);
      if(save_to_output){
        if(0 == get_local_id(0))
          *local_derivative_sum = 0.0;
        barrier(CLK_LOCAL_MEM_FENCE);

        int operation_index = get_local_id(0);
        while(operation_index < ==weight_relevant_operation_count==){
          *local_derivative_sum += operations_d_array[operation_index];
          operation_index += get_local_size(0);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(0 == get_local_id(0)){
          AtomicAdd(&d_w_array[d_w_index], *local_derivative_sum);
          AtomicAdd(&triggered_derivative_operations, ==weight_relevant_operation_count==);
        }
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }/*execute_derivative_workers()*/

    void execute_value_workers(
      int available_memory_slots, int weight_table_size, __constant double* network_inputs,
      __constant double* network_weights, __global double* operations_value_array, 
      __constant unsigned int* network_instruction_table, __constant unsigned int* neuron_index_to_spike_op_map
    ){
      ==operation_locals==
      const int operation_count = ==operation_count==;
      const unsigned int network_instruction_count = ==network_instruction_count==;
      int current_instruction_index = 0;
      while(current_instruction_index < network_instruction_count){
        if(0 == get_global_id(1)){
          int operations_to_do_now = network_instruction_table[current_instruction_index];
          int local_operation_index = 0; 
          while((local_operation_index + get_local_id(0)) < operations_to_do_now){
            int current_instruction_entry_start = current_instruction_index + 1 + ((local_operation_index + get_local_id(0)) * ==one_neural_instruction_entry_size==);
            __constant unsigned int* network_instructions = &network_instruction_table[current_instruction_entry_start];

            ==value_command_list_parsers==

            local_operation_index += get_local_size(0);
          }
          current_instruction_index += 1 + (operations_to_do_now * ==one_neural_instruction_entry_size==);
        }else{ // For any thread but the first in get_global_id(1), just skip the instructions
          current_instruction_index = network_instruction_count;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }/*execute_value_workers()*/

    void __kernel autodiff_iterate(
      __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
      __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      const long int neuron_index_to_spike_op_map_start_index = input_sizes[0] + input_sizes[1] + input_sizes[2];
      const long int network_instruction_table_start_index = neuron_index_to_spike_op_map_start_index + input_sizes[3];
      const long int meta_start_index = network_instruction_table_start_index + input_sizes[4];
      const long int weight_table_size = input_sizes[0];

      const int number_of_sequences = ==number_of_sequences==;
      const int minibatch_size = ==minibatch_size==;
      const int network_memory_size = ==network_memory_size==;
      const int sequence_inputs_count = ==prefill_num== + ==sequence_size==;
      const int raw_sequence_truncation = inputs[meta_start_index + 1];
      const int labels_in_sequence = ==sequence_size==;
      const int sequence_truncation = (raw_sequence_truncation == 0)?(labels_in_sequence):(max(1, raw_sequence_truncation));
      const int neuron_count = ==neuron_count==;
      const int operation_count = ==operation_count==;
      const int sequences_in_work_groups = (minibatch_size / get_num_groups(0)) + 1;
      const int d_w_threads_count = ==d_w_threads_count==;
      const int one_d_w_thread_buffer_size = number_of_sequences * labels_in_sequence * operation_count;

      uint local_seed = (uint)(inputs[min(get_global_id(0), (size_t)(input_sizes[0]))] * 100000.0);
      __local int sequence_start;
      __local int sequences_in_this_group;
      __local double local_derivative_sum[==d_w_threads_count==];
      if(0 == get_local_id(0)){
        /* Sequence starts from the given input, with some safeguards, and each group gets their own sequence based on their id */
        sequence_start = (int)(inputs[meta_start_index]);
        sequence_start = max( 0, min(sequence_start, (number_of_sequences - minibatch_size)) );
        sequence_start = sequence_start + (get_group_id(0) * sequences_in_work_groups);
        sequences_in_this_group = min( sequences_in_work_groups, (number_of_sequences - sequence_start) );

        /* In case there is no sequence truncation, all of the sequence elements will be considered when calculating the derivative */
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      int network_inputs_start_index = weight_table_size + sequence_start * ==one_input_size==;
      int network_labels_start_index = weight_table_size + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size==;
      int network_values_start_index = sequence_start * sequence_inputs_count * operation_count;

      if(0 == get_global_id(0)){ //reset d_w_array
        int weight_index = get_global_id(1);
        while(weight_index < weight_table_size){
          outputs[output_sizes[0] + output_sizes[1] + weight_index] = 0.0;
          weight_index += get_global_size(1);
        }
      }
      if(0 == get_global_id(0) && 0 == get_global_id(1)){
        triggered_derivative_operations = 0.0;
      }
      barrier(CLK_GLOBAL_MEM_FENCE);

      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        int network_derivatives_start_index = (
          output_sizes[0] // operations_value_array size
          + sequence_index * labels_in_sequence * operation_count // offset based on sequence index
        );
        int network_ran_count = 0;
        int available_memory_slots = 0;
        #pragma unroll
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          execute_value_workers(
            available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/, &outputs[network_values_start_index]/*operations_value_array*/,
            (__constant unsigned int*)&inputs[network_instruction_table_start_index]/*network_instruction_table*/,
            (__constant unsigned int*)&inputs[neuron_index_to_spike_op_map_start_index]/* neuron_index_to_spike_op_map */
          );
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          network_inputs_start_index += ==one_input_size==;
          network_values_start_index += operation_count;
        }/*for(prefill of the sequence)*/

        #pragma unroll
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          execute_value_workers(
            available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/,&outputs[network_values_start_index]/*operations_value_array*/,
            (__constant unsigned int*)&inputs[network_instruction_table_start_index]/*network_instruction_table*/,
            (__constant unsigned int*)&inputs[neuron_index_to_spike_op_map_start_index]/* neuron_index_to_spike_op_map */
          );
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          if(label_index < labels_in_sequence-1){
            network_inputs_start_index += ==one_input_size==;
            network_labels_start_index += ==one_label_size==;
            network_values_start_index += operation_count;
          }
        }/*for(every label inside the sequence)*/

        network_ran_count -= ==sequence_size==;
        available_memory_slots = min(network_ran_count, network_memory_size);
        network_inputs_start_index -= ==one_input_size== * (==sequence_size== - 1);
        network_labels_start_index -= ==one_label_size== * (==sequence_size== - 1);
        network_values_start_index -= operation_count * (==sequence_size== - 1);

        uint sequence_truncation_start = get_random_number(
          max(1, (labels_in_sequence - sequence_truncation)), &local_seed
        );

        #pragma unroll
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          int d_w_index = get_global_id(1);
          while(d_w_index < weight_table_size){
            execute_derivative_workers(
              d_w_index, min(available_memory_slots, label_index), weight_table_size,
              (
                ( label_index >= sequence_truncation_start )
                &&( label_index < (sequence_truncation_start + sequence_truncation) )
              ),
              &inputs[network_inputs_start_index]/*network_inputs*/,
              &inputs[network_labels_start_index]/*labels*/,
              &inputs[0]/*network_weights*/,
              &outputs[network_values_start_index]/*operations_value_array*/,
              &outputs[network_derivatives_start_index + one_d_w_thread_buffer_size * get_global_id(1)]/*operations_d_array*/,
              &outputs[output_sizes[0] + output_sizes[1]]/*d_w_array*/,
              (__constant unsigned int*)&inputs[network_instruction_table_start_index]/*network_instruction_table*/,
              (__constant unsigned int*)&inputs[neuron_index_to_spike_op_map_start_index]/*neuron_index_to_spike_op_map*/,
              &local_derivative_sum[get_global_id(1)]/*tmp value to store weight derivatives sum*/
            );
            d_w_index += get_global_size(1);
          }
          barrier(CLK_GLOBAL_MEM_FENCE);

          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          if(label_index < labels_in_sequence-1){
            network_inputs_start_index += ==one_input_size==;
            network_labels_start_index += ==one_label_size==;
            network_values_start_index += operation_count;
            network_derivatives_start_index += operation_count;
          }
        }/*for(every label inside the sequence)*/
      }/*for(every relevant sequence index)*/

      barrier(CLK_GLOBAL_MEM_FENCE);
      if(0 == get_global_id(0)){ //d_w_array correction for triggered operations
        int weight_index = get_global_id(1);
        while(weight_index < weight_table_size){
          outputs[output_sizes[0] + output_sizes[1] + weight_index] /= triggered_derivative_operations;
          weight_index += get_global_size(1);
        }
      }
    }/*kernel*/
  )";

  RFASSERT_LOG("Starting to split operations into workers..");
  m_neuralPropagationInstructions =
      generate_propagation_instructions(operations);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==value_command_list_parsers=="),
      generate_value_kernels("network_inputs", "network_weights",
                             "operations_value_array", "operation_count",
                             m_settings));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==derivative_command_list_parsers=="),
      generate_derivative_kernels("network_inputs", "labels", "network_weights",
                                  "operations_value_array",
                                  "operations_d_array", "operation_count",
                                  m_settings));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_memory_size=="),
      std::to_string(m_network.memory_size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==operation_count=="),
      std::to_string(operations.size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==neuron_count=="),
      std::to_string(m_network.neuron_array_size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==sequence_size=="),
      std::to_string(m_dataSet->get_sequence_size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==prefill_num=="),
      std::to_string(m_dataSet->get_prefill_inputs_number()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==number_of_sequences=="),
      std::to_string(m_dataSet->get_number_of_sequences()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==minibatch_size=="),
      std::to_string(std::min(m_settings.get_minibatch_size(),
                              m_dataSet->get_number_of_sequences())));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==weight_relevant_operation_count=="),
      std::to_string(weight_relevant_operation_count));

  std::string operation_locals;
  std::set<Autodiff_operations> declared_operations;
  for (std::uint32_t operation_index = 0u; operation_index < operations.size();
       ++operation_index) {
    Autodiff_operations operation_type =
        operations[operation_index]->get_type();
    if (declared_operations.find(operation_type) == declared_operations.end()) {
      if ((/* Only add the locals for this operation type once */
           (operation_type !=
            ad_operation_network_weight_regularization_feature) ||
           (declared_operations.find(ad_operation_network_feature) ==
            declared_operations.end())) &&
          (/* Because the two operations types share local variables */
           (operation_type != ad_operation_network_feature) ||
           (declared_operations.find(
                ad_operation_network_weight_regularization_feature) ==
            declared_operations.end()))) {
        operation_locals +=
            operations[operation_index]->local_declaration_operation();
      }
      declared_operations.insert(operation_type);
    }
  }
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==d_w_threads_count=="),
      std::to_string(get_d_w_threads_count()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==operation_locals=="), operation_locals);
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_instruction_count=="),
      std::to_string(m_neuralPropagationInstructions.size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==one_neural_instruction_entry_size=="),
      std::to_string(s_oneNeuralInstructionEntrySize));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==one_input_size=="),
      std::to_string(m_dataSet->get_input_size()));
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==one_label_size=="),
      std::to_string(m_dataSet->get_feature_size()));
  std::vector<std::vector<std::uint32_t>> operations_matrix =
      generate_operation_paralell_matrix(operations);
  std::uint32_t avg_row_size = 0u;
  for (const std::vector<std::uint32_t> &row : operations_matrix) {
    avg_row_size += row.size();
  }
  avg_row_size = static_cast<std::uint32_t>(
      std::ceil(static_cast<double>(avg_row_size) /
                static_cast<double>(operations_matrix.size())));
  m_maximumLocalWorkers = avg_row_size;
  m_builtSource = kernel_source;
  m_numberOfOperations = operations.size();
  m_built = true;
  RFASSERT_LOG("Build finished! One Local Group size: {}; \n Optimizer kernel "
               "source: \n {}",
               m_maximumLocalWorkers, m_builtSource);
}

} /* namespace rafko_gym */
