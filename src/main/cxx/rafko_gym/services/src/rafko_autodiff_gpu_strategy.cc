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

namespace rafko_gym{

void AutoDiffGPUStrategy::build(std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations){
  std::string source_base = R"(
    /* since network value needs to be calculated once per sequence item, it is in global memory */

    void execute_operation_index(
      operation_index,
      network_inputs, netowrk_input_size
      network_labels, network_labels_size
      calculated_values,
      calculated_derivatives,
      available_memory_slots
    ){
      // TODO: generate the operations
      switch (operation_index) {
        ==operations==
        case 0u:{

        }break;
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

  built = true;
}


} /* namespace rafko_gym */
