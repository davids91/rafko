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

#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_gym{

void AutoDiffGPUStrategy::build(std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations){
  std::string source_base = R"(
    void execute_value_operation(
      int operation_index, int available_memory_slots, int operation_number,
      __global double* network_inputs, int network_input_start,
      __global double* network_weights,
      __local double* operations_value_array, int operations_value_array_start
    ){
      switch (operation_index) {
        ==value_operations==
      }
    }

    void execute_derivative_operation(
      int operation_index, int available_memory_slots,
      __global double* network_inputs, int network_input_start,
      __global double* labels, int labels_start,
      __global double* network_weights,
      __local double* operations_value_array, int operations_value_array_start,
      __local double* operations_d_array, int operations_d_array_start
    ){
      switch (operation_index) {
        ==derivative_operations==
      }
    }

    void kernel autodiff_iterate(
      __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
      __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      const int network_memory_size = ==network_memory_size==;
      double derivative_values[network_memory_size][==operation_count==];
      double network_values[network_memory_size][==neuron_count==];

      //TODO: Calculate weights to do in one worker
      for(int sequence_index = ???; sequence_index < ???; ++sequence_index){
        int available_memory_slots = 0;
        for(int prefill_index = 0; prefill_index < ==prefill_num==; ++prefill_index){
          //TODO: Calculate value
          available_memory_slots = max( available_memory_slots + 1, network_memory_size );
        }
        for(int label_index = 0; label_index < ==label_num==; ++label_index){
          //TODO: Calculate value
          //TODO: calculate derivative
          //-->both value and derivative are based on local group id-s
          //-->calling execute_operation_index() based on the id-s
          //-->and based on which operations can be executed in paralell
          //TODO: average the relevant weight derivatives and udpdate the output buffer with them
          available_memory_slots = max( available_memory_slots + 1, network_memory_size );
        }
      }
    }/*kernel*/
  )";

  std::string value_operations;
  std::string derivative_operations;
  for(std::uint32_t operation_index = 0u; operation_index < operations.size(); ++operation_index){
    std::string operation = R"(
      case ==index==:{
        ==operation==
      }break;
    )";
    operation = rafko_utilities::replace_all_in_string(
      operation, std::regex("==index=="), std::to_string(operation_index)
    );
    operation = rafko_utilities::replace_all_in_string(
      operation, std::regex("==operation=="), operations[operation_index]->value_kernel_operation(
        "network_inputs","network_input_start", "network_weights","0",
        "operations_value_array","operations_value_array_start",
        std::to_string(operations.size()) /*operations_array_size*/
      )
    );
    value_operations += operation;
    operation = R"(
      case ==index==:{
        ==operation==
      }break;
    )";
    operation = rafko_utilities::replace_all_in_string(
      operation, std::regex("==index=="), std::to_string(operation_index)
    );
    //TODO: eliminate weight array start!
    operation = rafko_utilities::replace_all_in_string(
      operation, std::regex("==operation=="), operations[operation_index]->derivative_kernel_operation(
        "network_inputs", "network_input_start",
        "labels", "labels_start",
        "network_weights", "0",
        "operations_value_array", "operations_value_array_start",
        "operations_d_array", "operations_d_array_start",
        std::to_string(operations.size()) /*operations_array_size*/
      )
    );
    derivative_operations += operation;
  }

  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==value_operations=="), value_operations
  );
  source_base = rafko_utilities::replace_all_in_string(
    source_base, std::regex("==derivative_operations=="), derivative_operations
  );

  std::cout << "Optimizer source: " << source_base << std::endl;

  built = true;
}


} /* namespace rafko_gym */
