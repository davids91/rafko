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
#include <cmath>

#include "rafko_utilities/models/rafko_gpu_kernel_library.h"
#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_gym{

std::vector<rafko_mainframe::RafkoNBufShape> AutoDiffGPUStrategy::get_output_shapes() const{
  RFASSERT_LOG(
    "Autdif GPU Strategy output buffer overall size: ({} + {} + {})",
    /* operation values */
    (environment->get_number_of_sequences() * network.memory_size() * number_of_operations),
    ( /* operation derivatives */
      environment->get_number_of_sequences() * network.memory_size()
      * network.weight_table_size() * number_of_operations
    ),
    /* Weight derivatives */
    static_cast<std::uint32_t>(network.weight_table_size())
  );
  return{ rafko_mainframe::RafkoNBufShape{
    /* operation values */
    (environment->get_number_of_sequences() * network.memory_size() * number_of_operations),
    ( /* operation derivatives */
      environment->get_number_of_sequences() * network.memory_size()
      * network.weight_table_size() * number_of_operations
    ),
    /* Weight derivatives */
    static_cast<std::uint32_t>(network.weight_table_size())
  } };
}

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

void AutoDiffGPUStrategy::build(
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations,
  std::uint32_t weight_relevant_operation_count
){
  std::string source_base = rafko_utilities::atomic_double_average_function + rafko_utilities::random_function + R"(
    void execute_local_derivative_worker(
      int available_memory_slots, int weight_table_size, bool save_to_output,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array
    ){
      ==operation_locals==
      const int weights_in_one_thread = 1 + (weight_table_size / get_local_size(0));
      const int weights_index_start = weights_in_one_thread * get_local_id(0);
      const int weights_in_this_thread = min(
        weights_in_one_thread, max(0, (weight_table_size - weights_index_start))
      );
      // printf(
      //   "global[%d], local[%d / %d]: weights_in_one_thread: %d; weights_index_start: %d/%d; weights_in_this_thread: %d   \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)),
      //   weights_in_one_thread, weights_index_start, weight_table_size, weights_in_this_thread
      // );

      //TODO: index derivative arrays correctly : each weight to have their own array!
      for(int d_w_index = weights_index_start; d_w_index < (weights_index_start + weights_in_this_thread); ++d_w_index){
        const int operations_d_array_start = d_w_index * ==operation_count==;
        __global double* operations_d_array_for_w = &operations_d_array[operations_d_array_start];
        ==derivative_operations==

        // if(0 == get_local_id(0) && d_w_index == 6){
        //   int operation_index = 36;
        //   printf(
        //     "global[%d], local[%d / %d], weight[%d]: operation_d[%d] = %f\n",
        //     (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)), d_w_index,
        //     operation_index, operations_d_array_for_w[operation_index]
        //   );
        // }

        if(save_to_output){
          double average_gradient = 0.0;
          const double relevant_operation_count = ==weight_relevant_operation_count==;
          for(int operation_index = 0; operation_index < relevant_operation_count; ++operation_index){
            average_gradient += operations_d_array_for_w[operation_index];
          }
          average_gradient /= relevant_operation_count;
          AtomicAvg(&d_w_array[d_w_index], average_gradient);
        }
        // if(2 == get_local_id(0)){
        //   printf(
        //     "global[%d], local[%d / %d], weight[%d]: ",
        //     (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)), d_w_index
        //   );
        //   for(int operation_index = 30; operation_index < 37; ++operation_index){
        //     printf("[%f]", operations_d_array_for_w[d_w_index]);
        //   }
        //   printf("\n");
        // }/*if(debug)*/
      }/*for(all relevant weights)*/
    }/*execute_local_derivative_worker()*/

    void execute_local_value_worker(
      int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* network_weights,
      __global double* operations_value_array
    ){
      ==operation_locals==
      ==operation_switches==
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
      // if(0 == get_local_id(0)){
      //   printf("Shifting back local buffers!\n");
      // }
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
      /* deciding main parameters of the kernel */
      const int number_of_sequences = ==number_of_sequences==;
      const int minibatch_size = ==minibatch_size==;
      const int network_memory_size = ==network_memory_size==;
      const int operation_count = ==operation_count==;
      const int neuron_count = ==neuron_count==;

      /* deciding the sequence start index values for each workgroup */
      const int sequences_in_work_group = (number_of_sequences / get_num_groups(0)) + 1;
      uint local_seed = (uint)(inputs[min(get_global_id(0), (size_t)(input_sizes[0]))] * 100000.0);
      __local int sequence_start;
      __local int sequences_in_this_group;
      if(0 == get_local_id(0)){
        sequence_start = get_random_number(minibatch_size, &local_seed);
        sequences_in_this_group = min( sequences_in_work_group, (minibatch_size - sequence_start) );
      }
      work_group_barrier(CLK_LOCAL_MEM_FENCE);
      // printf(
      //   "global[%d], local[%d]: Number of sequences in this group: %d \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequences_in_this_group
      // );
      /* Main cache variables for running */
      int sequence_truncation = inputs[input_sizes[0] + input_sizes[1] + input_sizes[2]];
      int network_inputs_start_index = (input_sizes[0]/*weight_table_size*/ + sequence_start * ==one_input_size==);
      int network_labels_start_index = (input_sizes[0]/*weight_table_size*/ + input_sizes[1]/*network_inputs*/ + sequence_start * ==one_label_size==);
      int network_values_start_index = (sequence_start * network_memory_size * operation_count);
      int network_derivatives_start_index = (
        output_sizes[0] + sequence_start * network_memory_size * inputs[0]/*weight_table_size*/ * operation_count
      );

      // printf(
      //   "global[%d], local[%d]: weight_table_size: %d \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), input_sizes[0]
      // );
      // printf(
      //   "global[%d], local[%d]: sequence_start: %d \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequence_start
      // );
      // printf(
      //   "global[%d], local[%d]: one_input_size: %d \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), ==one_input_size==
      // );
      // printf(
      //   "global[%d], local[%d]: network_input initial start_index: %d \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), network_inputs_start_index
      // );
      /* In case there is no sequence truncation, all of the sequence elements will be considered when calculating the derivative */
      sequence_truncation = (sequence_truncation == 0)?(==sequence_size==):(sequence_truncation);
      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        // printf(
        //   "global[%d], local[%d]: ======================= sequence: %d (%d %d %d work groups)\n",
        //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequence_index, get_num_groups(0), get_num_groups(1), get_num_groups(2)
        // );
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
            available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/
          );
          ++network_ran_count;
          if(prefill_index < ==prefill_num==-1){
            available_memory_slots = min(network_ran_count, (network_memory_size-1));
            network_inputs_start_index += ==one_input_size==;
            if(network_ran_count < network_memory_size){
              network_values_start_index += operation_count;
            }else{
              shift_local_buffer_back(
                &outputs[network_values_sequence_start_index]/*operation_values*/,
                operation_count, available_memory_slots, false/*reset_last*/
              );
            }
          }
        }/*for(prefill of the sequence)*/
        uint sequence_truncation_start = get_random_number(
          max(1, (==sequence_size== - sequence_truncation)), &local_seed
        );
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          // printf(
          //   "global[%d], local[%d]: available memory: %d; network_ran_count: %d; network_memory_size: %d; label_in_sequence: %d/%d\n",
          //   (int)(get_global_id(0)), (int)(get_local_id(0)),
          //   available_memory_slots, network_ran_count, network_memory_size,
          //   label_index, ==sequence_size==
          // );
          execute_local_value_worker(
            available_memory_slots, input_sizes[0]/*weight_table_size*/,
            &inputs[network_inputs_start_index]/*network_inputs*/,
            &inputs[0]/*network_weights*/,
            &outputs[network_values_start_index]/*operation_values*/
          );
          // printf(
          //   "global[%d], local[%d]: label index: %d; seq_trun_start: %d; seq_trun_size: %d \n",
          //   (int)(get_global_id(0)), (int)(get_local_id(0)),
          //   label_index, sequence_truncation_start, sequence_truncation
          // );
          execute_local_derivative_worker(
            available_memory_slots, input_sizes[0]/*weight_table_size*/,
            (
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
          if(label_index < ==sequence_size==-1){
            available_memory_slots = min(network_ran_count, (network_memory_size-1));
            network_inputs_start_index += ==one_input_size==;
            network_labels_start_index += ==one_label_size==;
            if(network_ran_count < network_memory_size){
              network_values_start_index += operation_count;
              network_derivatives_start_index += operation_count * input_sizes[0]/*weight_table_size*/;
            }else{
              shift_local_buffer_back(
                &outputs[network_values_sequence_start_index]/*operation_values*/,
                operation_count, available_memory_slots, false/*reset_last*/
              );
              shift_local_buffer_back(
                &outputs[network_derivatives_sequence_start_index]/*operation_derivatives*/,
                operation_count * input_sizes[0]/*weight_table_size*/,
                available_memory_slots, false/*reset_last*/
              );
            }
          }
        }/*for(every label inside the sequence)*/
      }/*for(every relevant sequence index)*/
    }/*kernel*/
  )";
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==network_memory_size=="),
    std::to_string(network.memory_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_count=="),
    std::to_string(operations.size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==neuron_count=="),
    std::to_string(network.neuron_array_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==sequence_size=="),
    std::to_string(environment->get_sequence_size())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==prefill_num=="),
    std::to_string(environment->get_prefill_inputs_number())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==number_of_sequences=="),
    std::to_string(environment->get_number_of_sequences())
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==minibatch_size=="), std::to_string( std::min(
      settings.get_minibatch_size(), environment->get_number_of_sequences()
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
      operation_locals += operations[operation_index]->local_declaration_operation();
      declared_operations.insert(operation_type);
    }
  }
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_locals=="), operation_locals
  );

  std::vector<std::vector<std::uint32_t>> operations_matrix = generate_operation_paralell_matrix(operations);
  std::uint32_t avg_row_size = 0u;
  for(const std::vector<std::uint32_t>& row : operations_matrix) avg_row_size += row.size();
  avg_row_size = static_cast<std::uint32_t>( std::ceil(
    static_cast<double>(avg_row_size) / static_cast<double>(operations_matrix.size())
  ) );

  RFASSERT_LOG("Starting to split operations into workers..");
  std::string switch_case_template = R"(
    switch(get_local_id(0)){
      ==all_worker_cases==
      default:break;
    }
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  )";
  std::string worker_case_template = R"(
    case ==thread_index==:{
      ==thread_contents==
    }break;
  )";
  std::vector<std::string> value_phases(operations_matrix.size());
  std::uint32_t operations_phase = 0u;
  std::string all_switch_cases = "";
  for(const std::vector<std::uint32_t>& current_row : operations_matrix){
    /*!Note: one row to be split up between the workers */
    std::vector<std::string> worker_operations(avg_row_size);
    std::uint32_t placed_operation_count = 0u;
    while(placed_operation_count < current_row.size()){
      for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
        if((placed_operation_count + worker_index) < current_row.size()){
          std::uint32_t operation_index = current_row[placed_operation_count + worker_index];
          std::string operation = operations[operation_index]->value_kernel_operation(
            "network_inputs", "network_weights", "operations_value_array",
            std::to_string(operations.size()) /*operations_array_size*/
          );
          worker_operations[worker_index] += operation;
          RFASSERT_LOG(
            "Placing operation[current_row[{}] ==> {}] into worker[{}] phase[{}]",
            (placed_operation_count + worker_index), operation_index, worker_index, operations_phase
          );
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
      worker_case = rafko_utilities::replace_all_in_string(
        one_worker_case, std::regex("==thread_contents=="), worker_case
      );
      all_worker_cases += worker_case;
      ++worker_index;
    }
    all_switch_cases += rafko_utilities::replace_all_in_string(
      switch_case_template, std::regex("==all_worker_cases=="), all_worker_cases
    );
    RFASSERT_LOGV(
      current_row, "Used every available worker, row(at index {}/{}):",
      placed_operation_count, current_row.size()
    );
    ++operations_phase;
  }/*for(all rows in the operations_matrix)*/

  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==operation_switches=="), all_switch_cases
  );

  /* Add derivative operations into the kernel */
  std::string derivative_operations;
  /*!Note: Order is in reverse, because dependencies are pushed to the back of the array */
  for(std::int32_t operation_index = operations.size()-1; operation_index >= 0; --operation_index){
    derivative_operations += operations[operation_index]->derivative_kernel_operation(
      "network_inputs", "labels", "network_weights",
      "operations_value_array", "operations_d_array_for_w",
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
