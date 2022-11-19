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

#include <set>
#include <cmath>
#include <memory>
#include <atomic>

#include "rafko_utilities/models/rafko_gpu_kernel_library.hpp"
#include "rafko_utilities/services/rafko_string_utils.hpp"
#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym{

std::vector<rafko_mainframe::RafkoNBufShape> AutoDiffGPUStrategy::get_input_shapes() const{
  RFASSERT(static_cast<bool>(m_dataSet));
  RFASSERT_LOG(
    "Autodiff strategy Input shape: (weights: {} + inputs: {} + Labels: {} + sequence start: + sequence truncation: {} + d_w_index: {})",
    /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
    /* Inputs */ (m_dataSet->get_number_of_sequences() * m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
    /* Labels */(m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() * m_network.output_neuron_number()),
    /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u, /* d_w_index */ 1u
  );
  return{ rafko_mainframe::RafkoNBufShape{
    /* Weights */ (static_cast<std::uint32_t>(m_network.weight_table_size())),
    /* Inputs */ (m_dataSet->get_number_of_sequences() * m_dataSet->get_inputs_in_one_sequence() * m_network.input_data_size()),
    /* Labels */(m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() * m_network.output_neuron_number()),
    /* Sequence_start_index */ 1u, /* Sequence_truncation */ 1u, /* d_w_index */ 1u
  } };
}

std::vector<rafko_mainframe::RafkoNBufShape> AutoDiffGPUStrategy::get_output_shapes() const{
  RFASSERT_LOG(
    "Autdiff GPU Strategy output buffer overall size: (op values: {} + op derivatives: {} + w derivatives: {})",
    /* operation values */ (m_dataSet->get_number_of_sequences() * m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations ),
    /* operation derivatives */ (m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() * m_numberOfOperations),
    /* Weight derivatives */ static_cast<std::uint32_t>(m_network.weight_table_size())
  );
  return{ rafko_mainframe::RafkoNBufShape{
    ( /* operation values */
      m_dataSet->get_number_of_sequences() * m_dataSet->get_inputs_in_one_sequence() * m_numberOfOperations
    ),
    ( /* operation derivatives */
      m_dataSet->get_number_of_sequences() * m_dataSet->get_sequence_size() * m_numberOfOperations
    ),
    /* Weight derivatives */ static_cast<std::uint32_t>(m_network.weight_table_size())
  } };
}

std::vector<std::vector<std::uint32_t>> AutoDiffGPUStrategy::generate_operation_paralell_matrix(
  const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>& operations
){
  using DepthMarker = std::unique_ptr<std::atomic<std::uint32_t>>;

  const std::uint32_t number_of_threads = 8; /* Educated guess based on the average available CPU cores */
  rafko_utilities::ThreadGroup process_threads(number_of_threads);
  std::vector<DepthMarker> operations_depth;

  operations_depth.reserve(operations.size());
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    operations_depth.emplace_back(std::make_unique<std::atomic<std::uint32_t>>());
  }

  std::atomic<std::uint32_t> modified_operations{1}; /* not 0, to enter into the loop */
  std::atomic<std::uint32_t> max_depth{0};
  while(0 < modified_operations){
    modified_operations = 0;
    process_threads.start_and_block(
    [number_of_threads, &operations, &operations_depth, &max_depth, &modified_operations](std::uint32_t thread_index){
      const std::uint32_t operations_in_one_group = 1 + (operations.size() / number_of_threads);
      const std::uint32_t operations_start_index = operations_in_one_group * thread_index;
      const std::uint32_t operations_in_this_group = std::min(
        operations_in_one_group, static_cast<std::uint32_t>(
          std::max(0, static_cast<std::int32_t>(operations.size()) - static_cast<std::int32_t>(operations_start_index))
        )
      );
      for(std::uint32_t operation_index = operations_start_index; operation_index < (operations_start_index + operations_in_this_group); ++operation_index){
        for(RafkoBackpropagationOperation::Dependency& d : operations[operation_index]->get_dependencies()) {
          if(*operations_depth[operation_index] <= *operations_depth[d->get_operation_index()]){
            operations_depth[operation_index]->store(*operations_depth[d->get_operation_index()] + 1u);
            modified_operations.fetch_add(1u);
            max_depth.store( std::max(max_depth, *operations_depth[operation_index]) );
          }
        }/*for(every dependency in that operation)*/
      }/*for(every operation)*/
    });
  }/*while(there are modified operations)*/

  std::vector<std::vector<std::uint32_t>> operations_matrix(max_depth + 1);
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    operations_matrix[*operations_depth[operation_index]].push_back(operation_index);
  }

  RFASSERT_LOGV2(operations_matrix, "Operations matrix:");
  return operations_matrix;
}

std::string AutoDiffGPUStrategy::generate_switch_case_kernels_from(
  const std::vector<OperationsType>& operations,
  const std::vector<std::vector<std::uint32_t>>& operations_matrix,
  std::function<std::string(OperationsType)> operation_generator
){
  const static std::string switch_case_template = R"(
    ==multi_worker_operations==
    switch(get_local_id(0)){
      ==all_worker_cases==
      default:break;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  )";
  const static std::string worker_case_template = R"(
    case ==thread_index==:{
      ==thread_contents==
    }break;
  )";

  std::uint32_t operations_phase = 0u;
  std::uint32_t avg_row_size = 0u;
  std::string result_switch_cases;
  std::vector<std::string> value_phases(operations_matrix.size());
  for(const std::vector<std::uint32_t>& row : operations_matrix) avg_row_size += row.size();
  avg_row_size = static_cast<std::uint32_t>( std::ceil(
    static_cast<double>(avg_row_size) / static_cast<double>(operations_matrix.size())
  ) );
  m_maximumLocalWorkers = avg_row_size;

  for(const std::vector<std::uint32_t>& current_row : operations_matrix){
    /*!Note: one row to be split up between the workers */
    std::vector<std::string> worker_operations(avg_row_size);
    std::string multi_worker_operations = "";
    std::uint32_t placed_operation_count = 0u;
    while(placed_operation_count < current_row.size()){
      for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
        if((placed_operation_count + worker_index) < current_row.size()){
          std::uint32_t operation_index = current_row[placed_operation_count + worker_index];
          if(operations[operation_index]->is_multi_worker()){
            multi_worker_operations += operation_generator(operations[operation_index]);
            RFASSERT_LOG(
              "Placing operation[current_row[{}] ==> {}] before worker[{}] phase[{}](it's a multi-worker operation)",
              (placed_operation_count + worker_index), operation_index, worker_index, operations_phase
            );
          }else{
            worker_operations[worker_index] += operation_generator(operations[operation_index]);
            RFASSERT_LOG(
              "Placing operation[current_row[{}] ==> {}] into worker[{}] phase[{}]",
              (placed_operation_count + worker_index), operation_index, worker_index, operations_phase
            );
          }
        }
      }/*for(every worker thread)*/
      placed_operation_count += avg_row_size;
    }/*while(any operation remain unplaced)*/

    /* wrap every worker operation into a switch case*/
    std::uint32_t worker_index = 0;
    std::string all_worker_cases = "";
    for(std::string& worker_case : worker_operations){
      std::string one_worker_case = rafko_utilities::replace_all_in_string(
        worker_case_template, std::regex("==thread_index=="), std::to_string(worker_index)
      );
      if(0 < worker_case.size()){
        worker_case = rafko_utilities::replace_all_in_string(
          one_worker_case, std::regex("==thread_contents=="), worker_case
        );
        all_worker_cases += worker_case;
      }
      ++worker_index;
    }
    std::string one_switch_case = rafko_utilities::replace_all_in_string(
      switch_case_template, std::regex("==all_worker_cases=="), all_worker_cases
    );
    one_switch_case = rafko_utilities::replace_all_in_string(
      one_switch_case, std::regex("==multi_worker_operations=="), multi_worker_operations
    );
    result_switch_cases += one_switch_case;
    RFASSERT_LOGV(
      current_row, "Used every available worker, row(at index {}/{}):",
      placed_operation_count, current_row.size()
    );
    ++operations_phase;
  }/*for(all rows in the operations_matrix)*/
  return result_switch_cases;
}

void AutoDiffGPUStrategy::build(
  const std::vector<OperationsType>& operations,
  std::uint32_t weight_relevant_operation_count
){
  std::string source_base =
    rafko_utilities::atomic_double_add_function
    + rafko_utilities::atomic_double_average_function
    + rafko_utilities::random_function + R"(

    __constant bool evaluate_network = true;
    /*!Note: The only purpose of the autodiff operations is to train the network
     * so evaluation is always true in this context; Simply solving the network
     * is not strictly topic of it.
     */

    void execute_derivative_workers(
      int d_w_index, int available_memory_slots, int weight_table_size, int operation_count, bool save_to_output,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array
    ){
      __global static double triggered_derivative_operations = 0.0;
      if(0 == get_global_id(0))
        triggered_derivative_operations = 0;

      ==operation_locals==
      ==derivative_operations==
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
      __global double* operations_value_array
    ){
      ==operation_locals==
      ==operation_switches==
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
      const int d_w_index = inputs[input_sizes[0] + input_sizes[1] + input_sizes[2] + input_sizes[3] + input_sizes[4]];
      const int weight_table_size = input_sizes[0];
      const bool calculate_value = (0 == d_w_index);
      uint local_seed = (uint)(inputs[min(get_global_id(0), (size_t)(input_sizes[0]))] * 100000.0);
      __local int sequence_start;
      __local int sequence_truncation;
      __local int sequences_in_this_group;
      if(0 == get_local_id(0)){
        /* Sequence starts from the given input, with some safeguards, and each group gets their own based on their id */
        sequence_start = (int)(inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]]);
        sequence_start = max( 0, min(sequence_start, (number_of_sequences - minibatch_size)) );
        sequence_start = sequence_start + (get_group_id(0) * sequences_in_work_groups);
        sequences_in_this_group = min( sequences_in_work_groups, (number_of_sequences - sequence_start) );

        /* In case there is no sequence truncation, all of the sequence elements will be considered when calculating the derivative */
        sequence_truncation = inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]];
        sequence_truncation = (sequence_truncation == 0)?(sequence_labels_count):(max(1, sequence_truncation));
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      int network_inputs_start_index = weight_table_size + sequence_start * ==one_input_size==;
      int network_labels_start_index = weight_table_size + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size==;
      int network_values_start_index = sequence_start * sequence_inputs_count * operation_count;
      int network_derivatives_start_index = output_sizes[0] + sequence_start * sequence_labels_count * operation_count;

      #pragma unroll
      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        int network_ran_count = 0;
        int available_memory_slots = 0;
        #pragma unroll
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          if(calculate_value){
            execute_value_workers(
              available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
              &inputs[0]/*network_weights*/, &outputs[network_values_start_index]/*operation_values*/
            );
          }
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, (network_memory_size-1));
          network_inputs_start_index += ==one_input_size==;
          network_values_start_index += operation_count;
        }/*for(prefill of the sequence)*/
        uint sequence_truncation_start = get_random_number(
          max(1, (sequence_labels_count - sequence_truncation)), &local_seed
        );
        #pragma unroll
        for(int label_index = 0; label_index < sequence_labels_count; ++label_index){
          if(calculate_value){
            execute_value_workers(
              available_memory_slots, weight_table_size, &inputs[network_inputs_start_index]/*network_inputs*/,
              &inputs[0]/*network_weights*/,&outputs[network_values_start_index]/*operation_values*/
            );
          }
          /*Note: The available memory slots differ for derivatives becuase of the prefill,
           * and handling it is an open question: should the values be available despite the derivatives aren't?
           */
          execute_derivative_workers(
            d_w_index, min(available_memory_slots, label_index), weight_table_size,
            operation_count, (
              ( label_index >= sequence_truncation_start )
              &&( label_index < (sequence_truncation_start + sequence_truncation) )
            ),
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[network_labels_start_index]/*labels*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/,
            &outputs[output_sizes[0] + output_sizes[1]]/*d_w_array*/
          );
          ++network_ran_count;
          if(label_index < sequence_labels_count-1){
            available_memory_slots = min(network_ran_count, (network_memory_size-1));
            network_inputs_start_index += ==one_input_size==;
            network_labels_start_index += ==one_label_size==;
            network_values_start_index += operation_count;
            network_derivatives_start_index += operation_count;
          }
        }/*for(every label inside the sequence)*/
      }/*for(every relevant sequence index)*/
    }/*kernel*/
  )";
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==network_memory_size=="),
    std::to_string(m_network.memory_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_count=="),
    std::to_string(operations.size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==neuron_count=="),
    std::to_string(m_network.neuron_array_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==sequence_size=="),
    std::to_string(m_dataSet->get_sequence_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==prefill_num=="),
    std::to_string(m_dataSet->get_prefill_inputs_number())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==number_of_sequences=="),
    std::to_string(m_dataSet->get_number_of_sequences())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==minibatch_size=="), std::to_string( std::min(
      m_settings.get_minibatch_size(), m_dataSet->get_number_of_sequences()
    ) )
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==weight_relevant_operation_count=="),
    std::to_string(weight_relevant_operation_count)
  );

  std::string operation_locals;
  std::set<Autodiff_operations> declared_operations;
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    Autodiff_operations operation_type = operations[operation_index]->get_type();
    if(declared_operations.find(operation_type) == declared_operations.end()){
      if(( /* Only add the locals for this operation type once */
        (operation_type != ad_operation_network_weight_regularization_feature)
        ||(declared_operations.find(ad_operation_network_feature) == declared_operations.end())
      )&&( /* Because the two operations types share local variables */
        (operation_type != ad_operation_network_feature)
        ||(declared_operations.find(ad_operation_network_weight_regularization_feature) == declared_operations.end())
      )){
        operation_locals += operations[operation_index]->local_declaration_operation();
      }
      declared_operations.insert(operation_type);
    }
  }
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_locals=="), operation_locals
  );

  std::vector<std::vector<std::uint32_t>> operations_matrix = generate_operation_paralell_matrix(operations);

  RFASSERT_LOG("Starting to split operations into workers..");
  std::string value_operation_switch_cases = generate_switch_case_kernels_from(
    operations, operations_matrix, [&operations](OperationsType operation)->std::string{
      return operation->value_kernel_operation(
        "network_inputs", "network_weights", "operations_value_array",
        std::to_string(operations.size()) /*operations_array_size*/
      );
    }
  );
  std::string derivative_operation_switch_cases = generate_switch_case_kernels_from(
    operations, operations_matrix, [&operations](OperationsType operation)->std::string{
      return operation->derivative_kernel_operation(
        "network_inputs", "labels", "network_weights",
        "operations_value_array", "operations_d_array",
        std::to_string(operations.size()), "operation_count"
      );
    }
  );

  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_switches=="), value_operation_switch_cases
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==derivative_operations=="), derivative_operation_switch_cases
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==one_input_size=="), std::to_string(m_dataSet->get_input_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==one_label_size=="), std::to_string(m_dataSet->get_feature_size())
  );
  RFASSERT_LOG("Optimizer source: {}", source_base);
  m_builtSource = source_base;
  m_numberOfOperations = operations.size();
  m_built = true;
}


} /* namespace rafko_gym */
