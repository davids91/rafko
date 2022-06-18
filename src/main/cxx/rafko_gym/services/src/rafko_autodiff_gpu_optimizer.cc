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
#include "rafko_gym/services/rafko_autodiff_optimizer.h"

namespace rafko_gym{

void RafkoAutodiffGPUOptimizer::upload_weight_table(){
  RFASSERT_LOG("Uploading network weight table(size: {} bytes) to device..", (sizeof(double) * network.weight_table_size()));
  cl_int return_value = opencl_queue.enqueueWriteBuffer(
    gpu_phase.get_input_buffer(), CL_TRUE/*blocking*/, 0u/*offset*/,
    (sizeof(double) * network.weight_table_size())/*size*/,
    network.mutable_weight_table()->Mutable(0)
  );
  RFASSERT( return_value == CL_SUCCESS );
}

std::vector<cl::Event> update_inputs(){
  return environment->upload_inputs_to_buffer(
    opencl_queue, gpu_phase.get_input_buffer(),
    0u/*buffer_start_byte_offset*/,
    0u/*sequence_start_index*/, (
      sizeof(double) * static_cast<std::uint32_t>(network.weight_table_size())
    )/*buffer_sequence_start_index*/,
    environment->get_number_of_sequences()/*sequences_to_upload*/
  );
}

std::vector<cl::Event> update_labels(){
  return environment->upload_labels_to_buffer(
    opencl_queue, gpu_phase.get_input_buffer(),
    0u/*buffer_start_byte_offset*/,
    0u/*sequence_start_index*/, (
      sizeof(double) * static_cast<std::uint32_t>(network.weight_table_size())
      + (sizeof(double) * environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * network.input_data_size())
    )/*buffer_sequence_start_index*/,
    environment->get_number_of_sequences()/*sequences_to_upload*/,
    0u/*start_index_inside_sequence*/, environment->get_sequence_size()/*sequence_truncation*/
  );
}

void RafkoAutodiffGPUOptimizer::iterate(bool refresh_environment){
  upload_weight_table();

  if(refresh_environment){
    std::vector<cl::Event> input_events = update_inputs()
    std::vector<cl::Event> label_events = update_labels()
  }

  //TODO: Run Phase
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), gpu_phase.get_solution_space())
  );
  gpu_phase( enque_arguments );

  //TODO: Weight updates based on result data
  //TODO: Training and test set evaluation ?
}

} /* namespace rafko_gym */
