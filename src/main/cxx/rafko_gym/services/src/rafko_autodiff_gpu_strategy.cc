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

void AutoDiffGPUStrategy::build(std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations){
  std::string source_base = R"(
    void execute_local_worker(
      int local_id, int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array
    ){
      ==operation_locals==
      switch(local_id){
        ==local_worker_cases==
      }
      const int weights_in_one_thread = weight_table_size / get_local_size(0);
      const int weights_index_start = weights_in_one_thread * get_local_id(0);
      const int weights_in_this_thread = min(
        weights_in_one_thread, max(0, (weight_table_size - weights_index_start))
      );
      for(int d_w_index = weights_index_start; d_w_index < (weights_index_start + weights_in_this_thread); ++d_w_index){
        ==derivative_operations==
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
      int network_labels_start_index = (input_sizes[0]/*weight_table_size*/ + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size== );
      int network_values_start_index = (sequence_start * operation_count);
      int network_derivatives_start_index = (output_sizes[0] + sequence_start * inputs[0]/*weight_table_size*/ * operation_count);
      //TODO: Step index values forward, until network memory(or label number..) permits
      /*!Note: Index values are not stepped forward, instead the last available memory slot is added, and then data is moved backwards */
      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        int available_memory_slots = 0;
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          execute_local_worker(
            get_local_id(0), available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[network_labels_start_index]/*labels*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/
          );
          available_memory_slots = max(available_memory_slots + 1, network_memory_size);
          shift_local_buffer_back(
            &outputs[0]/*operation_values*/,
            operation_count,
            network_memory_size,
            false/*reset_last*/
          );
          shift_local_buffer_back(
            &outputs[output_sizes[0]]/*operation_derivatives*/,
            operation_count * input_sizes[0]/*weight_table_size*/,
            network_memory_size,
            false/*reset_last*/
          );
        }/*for(prefill of the sequence)*/
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          execute_local_worker(
            get_local_id(0), available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[network_labels_start_index]/*labels*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/,
            &outputs[network_derivatives_start_index]/*operation_derivatives*/
          );
          //TODO: average the relevant weight derivatives and udpdate the output buffer with them
          available_memory_slots = max(available_memory_slots + 1, network_memory_size);
          shift_local_buffer_back(
            &outputs[0]/*operation_values*/,
            operation_count,
            network_memory_size,
            false/*reset_last*/
          );
          shift_local_buffer_back(
            &outputs[output_sizes[0]]/*operation_derivatives*/,
            operation_count * input_sizes[0]/* weight_table_size */,
            network_memory_size,
            false/*reset_last*/
          );
        }
        //TODO: update the starting index values
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

  /* Find the operations that can be run simultaniously */
  std::uint32_t avg_row_size = 1u;
  std::uint32_t done_marker = operations.size();
  std::vector<bool> operation_included_mark(operations.size());
  std::vector<std::vector<std::uint32_t>> operations_matrix(1);
  /*!Note: the operations in each row can be executed in paralell;
   * Meaning all operations who have 'done' dependencies in an iteration
   * can be run in paralell in that iteration
   */
   //TODO: There are surely more, than 1 workers generated from 4 Neurons ans 36 operations...
   //TODO: default case where only barriers are present..
  while(0 < done_marker){ /* Until the whole operation array is processed */
    /* Collect all operations with max dependency_index pointing above marker */
    for(std::uint32_t operation_index = 0; operation_index < done_marker; ++operation_index){
      if(!operation_included_mark[operation_index]){
        std::uint32_t max_dependency_index = operations[operation_index]->get_max_dependency_index();
        bool no_unmarked_dependencies = false;
        if(max_dependency_index >= done_marker){
          /*!Note: Since done_marker starts at operations.size() and decreases,
           * the above line implicitly checks for operation size as well
           */
          no_unmarked_dependencies = true;
        }else{ /* check if all dependencies are marked */
          std::uint32_t marked_dependencies = 0u;
          std::vector<Dependency> dependencies = operations[operation_index]->get_dependencies();
          for(const Dependency& dependency : dependencies){
            if(operation_included_mark[dependency->get_operation_index()])
              ++marked_dependencies;
          }
          no_unmarked_dependencies = (
            (0u == dependencies.size())
            ||(marked_dependencies == dependencies.size())
          );
        }
        if(no_unmarked_dependencies){
          /*!Note: All operations not yet */
          operations_matrix.back().push_back(operation_index);
          operation_included_mark[operation_index] = true;
        }/*if(operation has all dependencies marked)*/
      }/*if(operation is not marked yet)*/
    }/*for(all operations below the marker)*/

    /* Decrease the marker until the first unmarked operation or the start of the array */
    while( (0 < done_marker)&&(operation_included_mark[done_marker - 1]) )
      --done_marker;

    /* Store the maximum length of the rows, and add a new row */
    avg_row_size = (avg_row_size + operations_matrix.back().size())/2u;
    operations_matrix.emplace_back();
  }

  /* For each thread create the network phases */
  std::string one_worker_case = " case 0:{ ==local_worker_phases== } break;";
  std::string one_worker_phase = R"(
    ==worker_operations==
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  )";

  std::vector<std::string> worker_phases(operations_matrix.size());
  /*!Note: one phase of workers involves some independent operations and a local barrier.
   * Because of coordination multiple barriers might be inserted into one worker case
   */
   /* Add value operations into worker phases */
  for(const std::vector<std::uint32_t>& operations_in_phase : operations_matrix){
    std::uint32_t placed_operation_count = 0u;
    while(placed_operation_count < operations_in_phase.size()){
      for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
        if((placed_operation_count + worker_index) < operations_in_phase.size()){
          std::string operation = operations[placed_operation_count + worker_index]->value_kernel_operation(
            "network_inputs", "network_weights", "operations_value_array",
            std::to_string(operations.size()) /*operations_array_size*/
          );
          worker_phases[worker_index] += operation;
        }
      }/*for(every worker thread)*/
      placed_operation_count += avg_row_size;
    }/*while(any operation remain unplaced)*/
    for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index)
      worker_phases[worker_index] += "\nwork_group_barrier(CLK_GLOBAL_MEM_FENCE);\n";

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
