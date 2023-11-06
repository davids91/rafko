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

namespace rafko_gym {

std::vector<rafko_mainframe::RafkoNBufShape>
AutoDiffGPUStrategy::get_input_shapes() const {
  RFASSERT(static_cast<bool>(m_dataSet));
  const std::size_t neural_propagation_instructions_size =
      (m_neuralPropagationInstructions.size() + 1u) /
      (sizeof(double) / sizeof(std::uint32_t));
  /*!Note: size + 1 is divided by the size ratio to make sure that there are
   * anough elements in the array */
  RFASSERT_LOG(
      "Autodiff strategy Input shape: (weights: {} + inputs: {} + Labels: {} + "
      "sequence start: + sequence truncation: {} + d_w_index: {} + Neural "
      "propagation instructions: {})",
      /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
      /* Inputs */
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
      /* Labels */
      (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_network.output_neuron_number()),
      /* Neural_propagation_instructions */
      neural_propagation_instructions_size,
      /*!Note: m_neuralPropagationInstructions has element type std::uint32_t,
       *not double! Because of the size difference Indexing has to be checked
       *carefully, and documented explicitly to help checking by-byte alignment!
       **/
      /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u,
      /* d_w_index */ 1u);
  return {rafko_mainframe::RafkoNBufShape{
      /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
      /* Inputs */
      (m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
      /* Labels */
      (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_network.output_neuron_number()),
      /* Neural_propagation_instructions */
      neural_propagation_instructions_size,
      /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u,
      /* d_w_index */ 1u}};
}

std::vector<rafko_mainframe::RafkoNBufShape>
AutoDiffGPUStrategy::get_output_shapes() const {
  RFASSERT_LOG("Autdiff GPU Strategy output buffer overall size: (op values: "
               "{} + op derivatives: {} + w derivatives: {})",
               /* operation values */
               (m_dataSet->get_number_of_sequences() *
                m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations),
               /* operation derivatives */
               (m_dataSet->get_number_of_sequences() *
                m_dataSet->get_sequence_size() * m_numberOfOperations),
               /* Weight derivatives */
               static_cast<std::uint32_t>(m_network.weight_table_size()));
  return {rafko_mainframe::RafkoNBufShape{
      (/* operation values */
       m_dataSet->get_number_of_sequences() *
       m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations),
      (/* operation derivatives */
       m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() *
       m_numberOfOperations),
      /* Weight derivatives */
      static_cast<std::uint32_t>(m_network.weight_table_size())}};
}

std::vector<std::vector<std::uint32_t>>
AutoDiffGPUStrategy::generate_operation_paralell_matrix(
    const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>
        &operations) {
  using DepthMarker = std::unique_ptr<std::atomic<std::uint32_t>>;

  const std::uint32_t number_of_threads =
      8; /* Educated guess based on the average available CPU cores */
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
      case ad_operation_network_input_d:
        RFASSERT(0u == operation->get_own_dependencies().size());
        result.insert(
            result.end(),
            {ad_operation_network_input_d,
             std::static_pointer_cast<RafkoBackpropNetworkInputOperation>(
                 operation)
                 ->get_weight_index(),
             std::static_pointer_cast<RafkoBackpropNetworkInputOperation>(
                 operation)
                 ->get_input_index(),
             0u, operation->get_operation_index(), 0u});
        break;
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
             upcasted_operation_ptr->get_own_dependencies_past_included()
                 .back()
                 ->get_operation_index(),
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
      "case ad_operation_network_input_d:{" +
      RafkoBackpropNetworkInputOperation::generic_value_kernel_operation(
          network_input_array, weight_array, operations_value_array) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_bias_d:{" +
      RafkoBackpropNeuronBiasOperation::generic_value_kernel_operation(
          weight_array, operations_value_array,
          "network_instructions[5]" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_input_d:{" +
      RafkoBackpropNeuronInputOperation::generic_value_kernel_operation(
          network_input_array, weight_array, operations_value_array,
          operations_array_size,
          "(network_instructions[5] & 0x00FFu)" /*behavior_index*/
          ) +
      "}break;";

  /* RafkoBackpropObjectiveOperation does not calculate value */

  kernel_source +=
      "case ad_operation_neuron_spike_d:{" +
      RafkoBackpropSpikeFnOperation::generic_value_kernel_operation(
          weight_array, operations_value_array, operations_array_size,
          "network_instructions[5]" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_transfer_d:{" +
      RafkoBackpropTransferFnOperation::generic_value_kernel_operation(
          operations_value_array, "network_instructions[5]" /*behavior_index*/,
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
      "case ad_operation_network_input_d:{" +
      RafkoBackpropNetworkInputOperation::generic_derivative_kernel_operation(
          network_input_array, operations_derivative_array) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_bias_d:{" +
      RafkoBackpropNeuronBiasOperation::generic_derivative_kernel_operation(
          weight_array, operations_value_array, operations_derivative_array,
          "network_instructions[5]" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_input_d:{" +
      RafkoBackpropNeuronInputOperation::generic_derivative_kernel_operation(
          network_input_array, weight_array, operations_value_array,
          operations_derivative_array, operations_array_size,
          "network_instructions[5]" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_objective_d:{" +
      RafkoBackpropObjectiveOperation::generic_derivative_kernel_operation(
          label_array, operations_value_array, operations_derivative_array,
          "network_instructions[5]" /*behavior_index*/,
          "==number_of_sequences==" /*sample_number*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_spike_d:{" +
      RafkoBackpropSpikeFnOperation::generic_derivative_kernel_operation(
          weight_array, operations_value_array, operations_derivative_array,
          operations_array_size, "network_instructions[5]" /*behavior_index*/
          ) +
      "}break;";
  kernel_source +=
      "case ad_operation_neuron_transfer_d:{" +
      RafkoBackpropTransferFnOperation::generic_derivative_kernel_operation(
          operations_value_array, operations_derivative_array,
          "network_instructions[5]" /*behavior_index*/, settings) +
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
      "((network_instructions[5] & 0xFF00u) >> 8)");
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
      kernel_source, std::regex("==weight_index=="),
      "network_instructions[1]");
  kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==dependency_descriptor=="),
      "network_instructions[3]");
}

void AutoDiffGPUStrategy::build(const std::vector<OperationsType> &operations,
                                std::uint32_t weight_relevant_operation_count) {
  std::string source_base = rafko_utilities::atomic_double_add_function +
                            rafko_utilities::atomic_double_average_function +
                            rafko_net::InputFunction::get_kernel_enums() +
                            rafko_net::TransferFunction::get_kernel_enums() +
                            rafko_net::SpikeFunction::get_kernel_enums() +
                            CostFunction::get_kernel_enums() +
                            RafkoBackpropagationOperation::get_kernel_enums() +
                            rafko_utilities::random_function + R"(

    __constant bool evaluate_network = true;
    /*!Note: The only purpose of the autodiff operations is to train the network
     * so evaluation is always true in this context; Simply solving the network
     * is not strictly topic of it.
     */

    void execute_derivative_workers(
      int d_w_index, int available_memory_slots, int weight_table_size, bool save_to_output,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array, 
      __constant unsigned int* network_instruction_table
    ){
      __global static double triggered_derivative_operations = 0.0;
      if(0 == get_global_id(0))
        triggered_derivative_operations = 0;

      ==operation_locals==
      const int operation_count = ==operation_count==;
      const unsigned int network_instruction_count = ==network_instruction_count==;
      int current_instruction_index = 0;
      while(current_instruction_index < network_instruction_count){
        int operations_to_do_now = network_instruction_table[current_instruction_index];
        int local_operation_index = 0; 
        while((local_operation_index + get_local_id(0)) < operations_to_do_now){
          int current_instruction_entry_start = current_instruction_index + 1 + ((local_operation_index + get_local_id(0)) * ==one_neural_instruction_entry_size==);
          __constant unsigned int* network_instructions = &network_instruction_table[current_instruction_entry_start];

          ==derivative_command_list_parsers== 

          local_operation_index += get_local_size(0);
        }
        current_instruction_index += 1 + (operations_to_do_now * ==one_neural_instruction_entry_size==);
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      barrier(CLK_GLOBAL_MEM_FENCE);
      if(save_to_output){
        #pragma unroll
        for(int operation_index = 0; operation_index < ==weight_relevant_operation_count==; ++operation_index){
          AtomicAdd(&d_w_array[d_w_index], operations_d_array[operation_index]);
        }
        if(0 == get_local_id(0))
          AtomicAdd(&triggered_derivative_operations, ==weight_relevant_operation_count==);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);

      if(0 == get_global_id(0) && 0 < triggered_derivative_operations){
        d_w_array[d_w_index] /= triggered_derivative_operations;
      }
    }/*execute_derivative_workers()*/

    void execute_value_workers(
      int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* network_weights,
      __global double* operations_value_array, 
      __constant unsigned int* network_instruction_table
    ){
      ==operation_locals==
      const int operation_count = ==operation_count==;
      const unsigned int network_instruction_count = ==network_instruction_count==;
      int current_instruction_index = 0;
      while(current_instruction_index < network_instruction_count){
        int operations_to_do_now = network_instruction_table[current_instruction_index];
        int local_operation_index = 0; 
        while((local_operation_index + get_local_id(0)) < operations_to_do_now){
          int current_instruction_entry_start = current_instruction_index + 1 + ((local_operation_index + get_local_id(0)) * ==one_neural_instruction_entry_size==);
          __constant unsigned int* network_instructions = &network_instruction_table[current_instruction_entry_start];

          ==value_command_list_parsers==

          local_operation_index += get_local_size(0);
        }
        current_instruction_index += 1 + (operations_to_do_now * ==one_neural_instruction_entry_size==);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }/*execute_value_workers()*/

    void __kernel autodiff_iterate(
      __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
      __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      const int number_of_sequences = ==number_of_sequences==;
      const int minibatch_size = ==minibatch_size==;
      const int network_memory_size = ==network_memory_size==;
      const int sequence_inputs_count = ==prefill_num== + ==sequence_size==;
      const int sequence_labels_count = ==sequence_size==;
      const int neuron_count = ==neuron_count==;
      const int operation_count = ==operation_count==;
      const int sequences_in_work_groups = (minibatch_size / get_num_groups(0)) + 1;
      const int d_w_index = inputs[input_sizes[0] + input_sizes[1] + input_sizes[2] + input_sizes[3] + input_sizes[4] + input_sizes[5]];
      const int weight_table_size = input_sizes[0];
      const bool calculate_value = (0 == d_w_index);
      uint local_seed = (uint)(inputs[min(get_global_id(0), (size_t)(input_sizes[0]))] * 100000.0);
      __local int sequence_start;
      __local int sequence_truncation;
      __local int sequences_in_this_group;
      if(0 == get_local_id(0)){
        /* Sequence starts from the given input, with some safeguards, and each group gets their own sequence based on their id */
        sequence_start = (int)(inputs[input_sizes[0] + input_sizes[1] + input_sizes[2] + input_sizes[3]]);
        sequence_start = max( 0, min(sequence_start, (number_of_sequences - minibatch_size)) );
        sequence_start = sequence_start + (get_group_id(0) * sequences_in_work_groups);
        sequences_in_this_group = min( sequences_in_work_groups, (number_of_sequences - sequence_start) );

        /* In case there is no sequence truncation, all of the sequence elements will be considered when calculating the derivative */
        sequence_truncation = inputs[input_sizes[0] + input_sizes[1] + input_sizes[2] + input_sizes[3] + input_sizes[4]];
        sequence_truncation = (sequence_truncation == 0)?(sequence_labels_count):(max(1, sequence_truncation));
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      int network_inputs_start_index = weight_table_size + sequence_start * ==one_input_size==;
      int network_labels_start_index = weight_table_size + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size==;
      int network_values_start_index = sequence_start * sequence_inputs_count * operation_count;
      int network_derivatives_start_index = output_sizes[0] + sequence_start * sequence_labels_count * operation_count;

      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        int network_ran_count = 0;
        int available_memory_slots = 0;
        #pragma unroll
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          if(calculate_value){
            execute_value_workers(
              available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
              &inputs[0]/*network_weights*/, &outputs[network_values_start_index]/*operation_values*/,
              (__constant unsigned int*)&inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]]/*network_instruction_table*/
            );
          }
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          network_inputs_start_index += ==one_input_size==;
          network_values_start_index += operation_count;
        }/*for(prefill of the sequence)*/
        uint sequence_truncation_start = get_random_number(
          max(1, (sequence_labels_count - sequence_truncation)), &local_seed
        );
        #pragma unroll
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          if(calculate_value){
            execute_value_workers(
              available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
              &inputs[0]/*network_weights*/,&outputs[network_values_start_index]/*operation_values*/,
              (__constant unsigned int*)&inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]]/*network_instruction_table*/
            );
          }
          /*Note: The available memory slots differ for derivatives becuase of the prefill,
           * and handling it is an open question: should the values be available despite the derivatives aren't?
           */
          execute_derivative_workers(
            d_w_index, min(available_memory_slots, label_index), weight_table_size,
            (
              ( label_index >= sequence_truncation_start )
              &&( label_index < (sequence_truncation_start + sequence_truncation) )
            ),
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[network_labels_start_index]/*labels*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/,
            &outputs[output_sizes[0] + output_sizes[1]]/*d_w_array*/,
            (__constant unsigned int*)&inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]]/*network_instruction_table*/
          );
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          if(label_index < sequence_labels_count-1){
            network_inputs_start_index += ==one_input_size==;
            network_labels_start_index += ==one_label_size==;
            network_values_start_index += operation_count;
            network_derivatives_start_index += operation_count;
          }
        }/*for(every label inside the sequence)*/
      }/*for(every relevant sequence index)*/
    }/*kernel*/
  )";

  RFASSERT_LOG("Starting to split operations into workers..");
  m_neuralPropagationInstructions =
      generate_propagation_instructions(operations);
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==value_command_list_parsers=="),
      generate_value_kernels("network_inputs", "network_weights",
                             "operations_value_array", "operation_count",
                             m_settings));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==derivative_command_list_parsers=="),
      generate_derivative_kernels("network_inputs", "labels", "network_weights",
                                  "operations_value_array",
                                  "operations_d_array", "operation_count",
                                  m_settings));

  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==network_memory_size=="),
      std::to_string(m_network.memory_size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==operation_count=="),
      std::to_string(operations.size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==neuron_count=="),
      std::to_string(m_network.neuron_array_size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==sequence_size=="),
      std::to_string(m_dataSet->get_sequence_size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==prefill_num=="),
      std::to_string(m_dataSet->get_prefill_inputs_number()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==number_of_sequences=="),
      std::to_string(m_dataSet->get_number_of_sequences()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==minibatch_size=="),
      std::to_string(std::min(m_settings.get_minibatch_size(),
                              m_dataSet->get_number_of_sequences())));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==weight_relevant_operation_count=="),
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
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==operation_locals=="), operation_locals);
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==network_instruction_count=="),
      std::to_string(m_neuralPropagationInstructions.size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==one_neural_instruction_entry_size=="),
      std::to_string(s_oneNeuralInstructionEntrySize));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==one_input_size=="),
      std::to_string(m_dataSet->get_input_size()));
  source_base = rafko_utilities::replace_all_in_string(
      source_base, std::regex("==one_label_size=="),
      std::to_string(m_dataSet->get_feature_size()));
  std::vector<std::vector<std::uint32_t>> operations_matrix =
      generate_operation_paralell_matrix(operations);
  std::uint32_t avg_row_size = 0u;
  for (const std::vector<std::uint32_t> &row : operations_matrix)
    avg_row_size += row.size();
  avg_row_size = static_cast<std::uint32_t>(
      std::ceil(static_cast<double>(avg_row_size) /
                static_cast<double>(operations_matrix.size())));
  m_maximumLocalWorkers = avg_row_size;
  m_builtSource = source_base;
  m_numberOfOperations = operations.size();
  m_built = true;
  RFASSERT_LOG("Build finished! One Local Group size: {}; \n Optimizer kernel "
               "source: \n {}",
               m_maximumLocalWorkers, m_builtSource);
}

} /* namespace rafko_gym */
