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
#include "rafko_gym/services/rafko_autodiff_gpu_strategy.h"

#include <set>

#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_gym{

std::vector<std::vector<std::uint32_t>> AutoDiffGPUStrategy::generate_operation_paralell_matrix(
  const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>& operations
){
  std::uint32_t done_marker = 0u;
  std::vector<bool> operation_included_mark(operations.size());
  std::vector<std::vector<std::uint32_t>> operations_matrix(1);

  /*!Note: the operations in each row can be executed in paralell;
   * Meaning all operations who have 'done' dependencies in an iteration
   * can be run in paralell in that iteration
   */
  std::vector<std::uint32_t> current_depth_dependencies;
  std::vector<std::uint32_t> next_depth_dependencies;
  std::uint32_t current_dependency_depth = 0u; /* 0 depth is in the last row of operations_matrix*/
  while(done_marker < operations.size()){
    RFASSERT(0u == next_depth_dependencies.size());

    while((done_marker < operations.size())&&(operation_included_mark[done_marker]))
      ++done_marker; /* Increase the number of done operations */

    RFASSERT_LOG("Done marker at: {}", done_marker);
    RFASSERT_LOG("current depth dependencies size: {}", current_depth_dependencies.size());
    if(0 == current_depth_dependencies.size()){ /* if no dependencies need to be extracted */
      current_dependency_depth = 0u;
      /* put the firstmost not yet included operation into the back of the matrix */
      if( (done_marker < operations.size())&&(!operation_included_mark[done_marker]) ){
        RFASSERT_LOG("Adding operation[{}] into current current_depth_dependencies", done_marker);
        operations_matrix.back().push_back(done_marker);
        current_depth_dependencies.push_back(done_marker);
        operation_included_mark[done_marker] = true;
      }
    }

    /* until there are no more of them, collect current depth dependencies into the next depth */
    RFASSERT_LOG("current depth dependencies grew to: {}", current_depth_dependencies.size());
    while(0u < current_depth_dependencies.size()){
      std::uint32_t dependencies_of = current_depth_dependencies.back();
      for(auto& d : operations[dependencies_of]->get_dependencies() ){
        std::uint32_t dependency_index = d->get_operation_index();
        if(!operation_included_mark[dependency_index]){
          RFASSERT_LOG("Pushing operation[{}] into next_depth_dependencies", dependency_index);
          next_depth_dependencies.push_back(dependency_index);
          operation_included_mark[dependency_index] = true;
        }
      }
      current_depth_dependencies.pop_back();
    }

    /* add additional layers into the operations_matrix if needed */
    ++current_dependency_depth;
    while(
      (0u < next_depth_dependencies.size())
      &&(operations_matrix.size() <= current_dependency_depth)
    ){
      RFASSERT_LOG("Adding depth {} to operations_matrix", current_dependency_depth);
      operations_matrix.insert(operations_matrix.begin(), std::vector<std::uint32_t>());
    }

    /* insert the dependencies into the target depth */
    RFASSERT_LOGV(next_depth_dependencies, "Adding operations to depth {}:", current_dependency_depth);
    for(std::uint32_t dependency_index : next_depth_dependencies){
      (operations_matrix.end() - current_dependency_depth - 1)->push_back(dependency_index);
    }

    /* The ol' switcheroo */
    RFASSERT(0u == current_depth_dependencies.size());
    current_depth_dependencies = next_depth_dependencies;
    next_depth_dependencies.clear();
    RFASSERT_LOGV(current_depth_dependencies,"current_depth_dependencies coming up:");
  }/*while(there are operations to put into the matrix yet)*/
  RFASSERT_LOGV2(operations_matrix, "Operations matrix:");
  return operations_matrix;
}

void AutoDiffGPUStrategy::build(std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations){
  std::string source_base = R"(

    void execute_local_derivative_worker(
      int local_id, int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array,
    ){
      ==operation_locals==
      //TODO: provide sequence truncation data
      const int weights_in_one_thread = weight_table_size / get_local_size(0);
      const int weights_index_start = weights_in_one_thread * get_local_id(0);
      const int weights_in_this_thread = min(
        weights_in_one_thread, max(0, (weight_table_size - weights_index_start))
      );
      for(int d_w_index = weights_index_start; d_w_index < (weights_index_start + weights_in_this_thread); ++d_w_index){
        ==derivative_operations==
        //TODO: average the relevant weight derivatives and udpdate the output buffer with them
        //TODO: ...in every non-truncated sequence
      }
    }

    void execute_local_value_worker(
      int local_id, int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* network_weights,
      __global double* operations_value_array
    ){
      ==operation_locals==
      switch(local_id){
        ==local_worker_cases==
        default:
        ==default_worker_case==
        break;
      }
    }

    void reset_buffer(__global double* buffer, int buffer_size){
      const int elements_per_worker = (buffer_size / get_local_size(0)) + 1;
      const int start_index = elements_per_worker * get_local_id(0);
      const int elements_in_this_worker = min(
        elements_per_worker, max((buffer_size - start_index), 0)
      );
      for(int i = 0; i < elements_in_this_worker; ++i){
        if((start_index + i) < buffer_size)
          buffer[start_index + i] = 0.0;
      }/*for(all elements to do in this worker)*/
      work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }

    void shift_local_buffer_back(__global double* buffer, int slot_size, int slot_number, bool reset_last){
      const int elements_per_worker = 1 + (slot_size / get_local_size(0));
      const int start_index_in_slot = elements_per_worker * get_local_id(0);
      const int elements_in_this_worker = min(
        elements_per_worker, max((slot_size - start_index_in_slot), 0)
      );
      if(0 < elements_in_this_worker){
        for(int slot_index = 0; slot_index < (slot_number - 1); ++slot_number){
          for(int i = 0; i < elements_in_this_worker; ++i){
            if((start_index_in_slot + i) < slot_size)
              buffer[(slot_index * slot_size) + (start_index_in_slot + i)] = (
                buffer[((slot_index + 1) * slot_size) + (start_index_in_slot + i)]
              );
          }/*for(all elements to do in this worker)*/
        }/*for(every slot in network memory)*/

        /* zero out latest slot */
        if(reset_last)reset_buffer(&buffer[(slot_number - 1) * slot_size], slot_size);
      }
      work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }

    void kernel autodiff_iterate(
      __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
      __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      const int network_memory_size = ==network_memory_size==;
      const int minibatch_size = ==minibatch_size==;
      const int operation_count = ==operation_count==;
      const int neuron_count = ==neuron_count==;
      const int sequences_in_work_group = (minibatch_size / get_num_groups(0)) + 1;
      const int sequence_start = sequences_in_work_group * get_local_id(0);
      const int sequences_in_this_group = min( sequences_in_work_group, (minibatch_size - sequence_start) );

      int network_inputs_start_index = (input_sizes[0]/*weight_table_size*/ + sequence_start * ==one_input_size==);
      int network_labels_start_index = (input_sizes[0]/*weight_table_size*/ + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size==);
      int network_values_start_index = (sequence_start * network_memory_size * operation_count);
      int network_derivatives_start_index = (
        output_sizes[0] + sequence_start * network_memory_size * inputs[0]/*weight_table_size*/ * operation_count
      );
      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        int network_ran_count = 0;
        int available_memory_slots = 0;
        network_values_start_index = (sequence_index * network_memory_size * operation_count);
        network_derivatives_start_index = (
          output_sizes[0] + sequence_index * network_memory_size * inputs[0]/*weight_table_size*/ * operation_count
        );
        int network_derivatives_sequence_start_index = network_derivatives_start_index;
        int network_values_sequence_start_index = network_values_start_index;
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          execute_local_value_worker(
            get_local_id(0), available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
          );
          ++network_ran_count;
          available_memory_slots = min(network_ran_count, network_memory_size);
          network_inputs_start_index += ==one_input_size==;
          if(network_ran_count < network_memory_size){
            network_values_start_index += operation_count;
          }else{
            shift_local_buffer_back(
              &outputs[network_values_sequence_start_index]/*operation_values*/,
              operation_count,
              network_memory_size,
              false/*reset_last*/
            );
          }
        }/*for(prefill of the sequence)*/
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          execute_local_value_worker(
            get_local_id(0), available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/
          );
          execute_local_derivative_worker(
            get_local_id(0), available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[network_labels_start_index]/*labels*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/,
            &outputs[output_sizes[0] + output_sizes[1]]/*d_w_array*/
          );
          network_inputs_start_index += ==one_input_size==;
          network_labels_start_index += ==one_label_size==;
          if(network_ran_count < network_memory_size){
            network_values_start_index += operation_count;
          }else{
            shift_local_buffer_back(
              &outputs[network_values_sequence_start_index]/*operation_values*/,
              operation_count,
              network_memory_size,
              false/*reset_last*/
            );
          }
          if(network_ran_count < network_memory_size){
            network_derivatives_start_index += operation_count * input_sizes[0]/*weight_table_size*/;
          }else{
            shift_local_buffer_back(
              &outputs[network_derivatives_sequence_start_index]/*operation_derivatives*/,
              operation_count * input_sizes[0]/*weight_table_size*/,
              network_memory_size,
              false/*reset_last*/
            );
          }
        }/*for(every label inside the sequence)*/
      }/*for(every relevant sequence index)*/
    }/*kernel*/
  )";
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==network_memory_size=="), std::to_string(network.memory_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_count=="), std::to_string(operations.size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==neuron_count=="), std::to_string(network.neuron_array_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==sequence_size=="), std::to_string(environment->get_sequence_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==prefill_num=="), std::to_string(environment->get_prefill_inputs_number())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==minibatch_size=="), std::to_string(used_minibatch_size)
  );

  std::string operation_locals;
  std::set<Autodiff_operations> declared_operations;
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    Autodiff_operations operation_type = operations[operation_index]->get_type();
    if(declared_operations.find(operation_type) == declared_operations.end()){
      operation_locals += operations[operation_index]->local_declaration_operation();
      declared_operations.insert(operation_type);
    }
  }
  std::vector<std::vector<std::uint32_t>> operations_matrix = generate_operation_paralell_matrix(operations);
  std::uint32_t avg_row_size = 0u;
  for(const std::vector<std::uint32_t>& row : operations_matrix) avg_row_size += row.size();
  avg_row_size /= operations_matrix.size();

  /* For each thread create the network phases */
  std::vector<std::string> worker_phases(operations_matrix.size());
  std::string default_worker_case = "";
  /*!Note: one phase of workers involves some independent operations and a local barrier.
   * Because of coordination multiple barriers might be inserted into one worker case
   */
   /* Add value operations into worker phases */
  RFASSERT_LOG("Starting to split operations into workers..");
  for(const std::vector<std::uint32_t>& operations_in_phase : operations_matrix){
    std::uint32_t placed_operation_count = 0u;
    while(placed_operation_count < operations_in_phase.size()){
      for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
        if((placed_operation_count + worker_index) < operations_in_phase.size()){
          std::uint32_t operation_index = operations_in_phase[placed_operation_count + worker_index];
          std::string operation = operations[operation_index]->value_kernel_operation(
            "network_inputs", "network_weights", "operations_value_array",
            std::to_string(operations.size()) /*operations_array_size*/
          );
          if(0 == operation_index)
            std::cout << "Operation[0]: " << operation << std::endl;
          worker_phases[worker_index] += operation;
          RFASSERT_LOG(
            "Placing operation[operation_in_phase[{}] ==> {}] into worker[{}] phase",
            (placed_operation_count + worker_index), operation_index, worker_index
          );
        }
      }/*for(every worker thread)*/
      placed_operation_count += avg_row_size;
    }/*while(any operation remain unplaced)*/
    for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index)
      worker_phases[worker_index] += "\nwork_group_barrier(CLK_GLOBAL_MEM_FENCE);\n";
    default_worker_case += "\nwork_group_barrier(CLK_GLOBAL_MEM_FENCE);\n";
  }/*for(every operation phase)*/
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_locals=="), operation_locals
  );
  std::string all_worker_cases;

  for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
    std::string one_worker_case = rafko_utilities::replace_all_in_string(
      "case ==op_index==:{==worker_operations==} break;\n",
      std::regex("==op_index=="), std::to_string(worker_index)
    );
    all_worker_cases += rafko_utilities::replace_all_in_string(
      one_worker_case, std::regex("==worker_operations=="), worker_phases[worker_index]
    );
  }
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==local_worker_cases=="), all_worker_cases
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==default_worker_case=="), default_worker_case
  );

  /* Add derivative operations into the kernel */
  std::string derivative_operations;
  /*!Note: Order is in reverse, because dependencies are pushed to the back of the array */
  for(std::int32_t operation_index = operations.size()-1; operation_index >= 0; --operation_index){
    derivative_operations += operations[operation_index]->derivative_kernel_operation(
      "network_inputs", "labels", "network_weights",
      "operations_value_array", "operations_d_array",
      std::to_string(operations.size()) /*operations_array_size*/
    );
  }
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==derivative_operations=="), derivative_operations
  );

  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==one_input_size=="), std::to_string(environment->get_input_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==one_label_size=="), std::to_string(environment->get_feature_size())
  );
  std::cout << "Optimizer source: " << source_base << std::endl;
  built_source = source_base;
  number_of_operations = operations.size();
  maximum_local_workers = avg_row_size;
  built = true;
}


} /* namespace rafko_gym */
