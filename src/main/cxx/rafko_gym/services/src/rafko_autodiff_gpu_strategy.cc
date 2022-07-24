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
#include <memory>
#include <atomic>

#include "rafko_utilities/models/rafko_gpu_kernel_library.h"
#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_gym{

std::vector<rafko_mainframe::RafkoNBufShape> AutoDiffGPUStrategy::get_input_shapes() const{
  RFASSERT(static_cast<bool>(environment));
  RFASSERT_LOG(
    "Autodiff strategy Input shape: {} + {} + {} + {}",
    /* Weights */ (static_cast<std::uint32_t>(network.weight_table_size())),
    /* Inputs */ (environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * network.input_data_size()),
    /* Labels */(environment->get_number_of_sequences() * environment->get_sequence_size() * network.output_neuron_number()),
    /* Sequence_truncation */ 1u
  );
  return{ rafko_mainframe::RafkoNBufShape{
    /* Weights */ (static_cast<std::uint32_t>(network.weight_table_size())),
    /* Inputs */ (environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * network.input_data_size()),
    /* Labels */(environment->get_number_of_sequences() * environment->get_sequence_size() * network.output_neuron_number()),
    /* Sequence_truncation */ 1u
  } };
}

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
    // RFASSERT_LOGV(
    //   operations_depth, "Modified operations: [{}]; operations depth(max:{}): ",
    //   modified_operations, max_depth
    // );
  }/*while(there are modified operations)*/

  //TODO: Multithread this ? maybe?
  std::vector<std::vector<std::uint32_t>> operations_matrix(max_depth + 1);
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    operations_matrix[*operations_depth[operation_index]].push_back(operation_index);
  }

  RFASSERT_LOGV2(operations_matrix, "Operations matrix:");
  return operations_matrix;
}

void AutoDiffGPUStrategy::build(
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations,
  std::uint32_t weight_relevant_operation_count
){ //TODO: Include "evaluate_network" into kernels
  std::string source_base =
    rafko_utilities::atomic_double_add_function
    + rafko_utilities::atomic_double_average_function
    + rafko_utilities::random_function + R"(

    __constant bool evaluate_network = true;
    /*!Note: The only purpose of the autodiff operations is to train the network
     * so evaluation is always true in this context; Simply solving the network
     * is not strictly topic of it.
     */

    void execute_local_derivative_worker(
      int available_memory_slots, int weight_table_size, int derivative_operation_count, bool save_to_output,
      __constant double* network_inputs, __constant double* labels, __constant double* network_weights,
      __global double* operations_value_array, __global double* operations_d_array, __global double* d_w_array
    ){
      ==operation_locals==
      const int weights_in_one_thread = 1 + (weight_table_size / get_local_size(0));
      const int weights_index_start = weights_in_one_thread * get_local_id(0);
      const int weights_in_this_thread = min(
        weights_in_one_thread, max(0, (weight_table_size - weights_index_start))
      );

      __global static double divisor = 0.0;
      // printf(
      //   "global[%d], local[%d / %d]: weights_in_one_thread: %d; weights_index_start: %d/%d; weights_in_this_thread: %d   \n",
      //   (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)),
      //   weights_in_one_thread, weights_index_start, weight_table_size, weights_in_this_thread
      // );

      for(int d_w_index = weights_index_start; d_w_index < (weights_index_start + weights_in_this_thread); ++d_w_index){
        __global double* operations_d_array_for_w = &operations_d_array[d_w_index * ==operation_count==];

        ==derivative_operations==

        // if(0 == get_local_id(0) && d_w_index == 6){
        //   int operation_index = 34;
        //   printf(
        //     "global[%d], local[%d / %d], weight[%d]: operation_d[%d] = %f\n",
        //     (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)), d_w_index,
        //     operation_index, operations_d_array_for_w[operation_index]
        //   );
        // }
        // if(2 == d_w_index){
        //   printf(
        //     "global[%d], local[%d / %d], weight[%d] derivatives:\n",
        //     (int)(get_global_id(0)), (int)(get_local_id(0)), (int)(get_local_size(0)), d_w_index
        //   );
        //   for(int operation_index = 0; operation_index < 37; ++operation_index){
        //     if(0 == (operation_index % 10))printf("\n");
        //     printf("[%10.10f]", operations_d_array_for_w[operation_index]);
        //   }
        //   printf("\n");
        // }/*if(debug)*/
        if(save_to_output){
          double average_gradient = 0.0;
          for(int operation_index = 0; operation_index < ==weight_relevant_operation_count==; ++operation_index){
            average_gradient += operations_d_array_for_w[operation_index];
          }
          // average_gradient /= ==weight_relevant_operation_count==.0;
          AtomicAdd(&d_w_array[d_w_index], average_gradient);
          AtomicAdd(&divisor, ==weight_relevant_operation_count==.0);
          //TODO: set gradient collection precision to be adjustable:
          //-->if big values are present in the gradients, then average of average;
          //-->if not, then sum / overall_count
        }
      }/*for(all relevant weights)*/

      work_group_barrier(CLK_GLOBAL_MEM_FENCE);
      const int elements_per_worker = (weight_table_size / get_local_size(0)) + 1;
      const int start_index = elements_per_worker * get_local_id(0);
      const int elements_in_this_worker = min(
        elements_per_worker, max((weight_table_size - start_index), 0)
      );
      for(int i = 0; i < elements_in_this_worker; ++i){
        if((start_index + i) < weight_table_size)
          d_w_array[start_index + i] /= divisor;
      }/*for(all elements to do in this worker)*/
      work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }/*execute_local_derivative_worker()*/

    void execute_local_value_worker(
      int available_memory_slots, int weight_table_size,
      __constant double* network_inputs, __constant double* network_weights,
      __global double* operations_value_array
    ){
      ==operation_locals==
      ==operation_switches==
    }/*execute_local_value_worker()*/

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
        for(int slot_index = 0; slot_index < (slot_number - 1); ++slot_index){
          // if(0 == get_local_id(0))printf(
          //   "global[%d], local[%d]: slot[%d / %d] --> element_count: %d \n",
          //   (int)(get_global_id(0)), (int)(get_local_id(0)),
          //   slot_index, slot_number, elements_in_this_worker
          // );

          for(int i = 0; i < elements_in_this_worker; ++i){
            if((start_index_in_slot + i) < slot_size){
              // if(0 == get_local_id(0))printf(
              //   "global[%d], local[%d]: buf[%d] = buf[%d] \n",
              //   (int)(get_global_id(0)), (int)(get_local_id(0)),
              //   (slot_index * slot_size) + (start_index_in_slot + i),
              //   ((slot_index + 1) * slot_size) + (start_index_in_slot + i)
              // );
              buffer[(slot_index * slot_size) + (start_index_in_slot + i)] = (
                buffer[((slot_index + 1) * slot_size) + (start_index_in_slot + i)]
              );
            }
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
      const int neuron_count = ==neuron_count==;
      const int operation_count = ==operation_count==;
      const int derivative_operation_count = input_sizes[0]/*weight_table_size*/ * operation_count;

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
        output_sizes[0] + sequence_start * network_memory_size * derivative_operation_count
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
      sequence_truncation = (sequence_truncation == 0)?(==sequence_size==):(max(1, sequence_truncation));
      #pragma unroll
      for(int sequence_index = sequence_start; sequence_index < (sequence_start + sequences_in_this_group); ++sequence_index){
        // printf(
        //   "global[%d], local[%d]: ======================= sequence: %d (%d %d %d work groups)\n",
        //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequence_index, get_num_groups(0), get_num_groups(1), get_num_groups(2)
        // );
        int network_ran_count = 0;
        int available_memory_slots = 0;
        network_values_start_index = (sequence_index * network_memory_size * operation_count);
        network_derivatives_start_index = (
          output_sizes[0] + sequence_index * network_memory_size * derivative_operation_count
        );
        int network_derivatives_sequence_start_index = network_derivatives_start_index;
        int network_values_sequence_start_index = network_values_start_index;
        #pragma unroll
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
                operation_count, network_memory_size, false/*reset_last*/
              );
            }
          }
        }/*for(prefill of the sequence)*/
        uint sequence_truncation_start = get_random_number(
          max(1, (==sequence_size== - sequence_truncation)), &local_seed
        );
        #pragma unroll
        for(int label_index = 0; label_index < ==sequence_size==; ++label_index){
          // printf(
          //   "global[%d], local[%d]: sequence[%d]; label[%d]; input_start: %d / %d; label_start %d / %d \n",
          //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequence_index, label_index,
          //   network_inputs_start_index, (input_sizes[0] + input_sizes[1]),
          //   network_labels_start_index, (input_sizes[0] + input_sizes[1] + input_sizes[2])
          // );
          // printf(
          //   "global[%d], local[%d]: sequence[%d]; label[%d]; memory: %d; value_start(%d): %d (+%d)/ %d; d_start(%d): %d(+%d) / %d \n",
          //   (int)(get_global_id(0)), (int)(get_local_id(0)), sequence_index, label_index, network_memory_size,
          //   network_values_sequence_start_index, network_values_start_index, operation_count, output_sizes[0],
          //   network_derivatives_sequence_start_index, network_derivatives_start_index, derivative_operation_count, (output_sizes[0] + output_sizes[1])
          // );
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
            derivative_operation_count, (
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
              network_derivatives_start_index += derivative_operation_count;
            }else{
              shift_local_buffer_back(
                &outputs[network_values_sequence_start_index]/*operation_values*/,
                operation_count, network_memory_size, false/*reset_last*/
              );
              shift_local_buffer_back(
                &outputs[network_derivatives_sequence_start_index]/*operation_derivatives*/,
                derivative_operation_count, network_memory_size, false/*reset_last*/
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
  std::uint32_t avg_row_size = 0u;
  for(const std::vector<std::uint32_t>& row : operations_matrix) avg_row_size += row.size();
  avg_row_size = static_cast<std::uint32_t>( std::ceil(
    static_cast<double>(avg_row_size) / static_cast<double>(operations_matrix.size())
  ) );

  RFASSERT_LOG("Starting to split operations into workers..");
  std::string switch_case_template = R"(
    ==multi_worker_operations==
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
    std::string multi_worker_operations = "";
    std::uint32_t placed_operation_count = 0u;
    while(placed_operation_count < current_row.size()){
      for(std::uint32_t worker_index = 0u; worker_index < avg_row_size; ++worker_index){
        if((placed_operation_count + worker_index) < current_row.size()){
          std::uint32_t operation_index = current_row[placed_operation_count + worker_index];
          std::string operation = operations[operation_index]->value_kernel_operation(
            "network_inputs", "network_weights", "operations_value_array",
            std::to_string(operations.size()) /*operations_array_size*/
          );
          if(operations[operation_index]->is_multi_worker()){
            multi_worker_operations += operation;
            RFASSERT_LOG(
              "Placing operation[current_row[{}] ==> {}] before worker[{}] phase[{}](it's a multi-worker operation)",
              (placed_operation_count + worker_index), operation_index, worker_index, operations_phase
            );
          }else{
            worker_operations[worker_index] += operation;
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
    all_switch_cases += one_switch_case;
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
      std::to_string(operations.size()) /*operations_array_size*/,
      "derivative_operation_count"
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
