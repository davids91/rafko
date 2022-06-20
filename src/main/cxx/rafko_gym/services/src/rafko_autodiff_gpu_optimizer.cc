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
#include "rafko_gym/services/rafko_autodiff_gpu_optimizer.h"

namespace rafko_gym{

void RafkoAutodiffGPUOptimizer::build(std::shared_ptr<RafkoObjective> objective){
  RFASSERT_SCOPE(AUTODIFF_GPU_BUILD);
  strategy->build(operations, build_without_data(objective));
  gpu_phase.set_strategy(strategy);
}

void RafkoAutodiffGPUOptimizer::upload_weight_table(){
  RFASSERT_LOG("Uploading network weight table(size: {} bytes) to device..", (sizeof(double) * network.weight_table_size()));
  cl_int return_value = opencl_queue.enqueueWriteBuffer(
    gpu_phase.get_input_buffer(), CL_TRUE/*blocking*/, 0u/*offset*/,
    (sizeof(double) * network.weight_table_size())/*size*/,
    network.mutable_weight_table()->Mutable(0)
  );
  RFASSERT( return_value == CL_SUCCESS );
}

std::vector<cl::Event> RafkoAutodiffGPUOptimizer::update_inputs(){
  return environment->upload_inputs_to_buffer(
    opencl_queue, gpu_phase.get_input_buffer(),
    0u/*buffer_start_byte_offset*/,
    0u/*sequence_start_index*/, (
      sizeof(double) * static_cast<std::uint32_t>(network.weight_table_size())
    )/*buffer_sequence_start_index*/,
    environment->get_number_of_sequences()/*sequences_to_upload*/
  );
}

std::vector<cl::Event> RafkoAutodiffGPUOptimizer::update_labels(){
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

void RafkoAutodiffGPUOptimizer::refresh_environment(){
  std::vector<cl::Event> input_events = update_inputs();
  std::vector<cl::Event> label_events = update_labels();
  for(cl::Event& e : input_events){
    cl_int return_value = e.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }
  for(cl::Event& e : label_events){
    cl_int return_value = e.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }
}

void RafkoAutodiffGPUOptimizer::iterate(bool refresh_environment){
  RFASSERT_SCOPE(AUTODIFF_GPU_ITERATE);
  upload_weight_table();
  if(refresh_environment)update_context_errors();

  std::thread tmp_reset_thread([this](){
    std::fill(tmp_avg_derivatives.begin(), tmp_avg_derivatives.end(), 0.0);
  });
  //TODO: Run for all weights
  gpu_phase();
  tmp_reset_thread.join();
  RFASSERT_LOG("sequence count: {}", environment->get_number_of_sequences());
  RFASSERT_LOG("network memory size: {}", network.memory_size());
  RFASSERT_LOG("operations count: {}", operations.size());
  RFASSERT_LOG("weights count: {}", network.weight_table_size());
  RFASSERT_LOG(
    "Getting Autodiff Phase weight derivatives({} numbers) from: [{}]",
    tmp_avg_derivatives.size(),
    ( /* operation values + operation derivatives size */
      (environment->get_number_of_sequences() * network.memory_size() * operations.size())
      + ( network.memory_size() * environment->get_number_of_sequences() * operations.size() * network.weight_table_size() )
    )/*offset*/
  );
  gpu_phase.load_output(
    tmp_avg_derivatives.data()/*target*/, tmp_avg_derivatives.size()/*size*/,
    ( /* operation values + operation derivatives size */
      (environment->get_number_of_sequences() * network.memory_size() * operations.size())
      + ( network.memory_size() * environment->get_number_of_sequences() * operations.size() * network.weight_table_size() )
    )/*offset*/
  );

  /*!Debug: See values output:*/
  std::vector<double> output_buffer_values(
    environment->get_number_of_sequences()
    * network.memory_size()
    * operations.size()
  );
  gpu_phase.load_output(
    output_buffer_values.data()/*target*/,
    output_buffer_values.size()/*size*/,
    0/*offset*/
  );

  std::cout << "output_values:";
  std::uint32_t i = 0;
  for(const double& d : output_buffer_values){
    if(0 == (i % operations.size()))std::cout << std::endl;
    if(0 == (i++ % 5))std::cout << " ";
    std::cout << "[" << d << "]";
  }
  std::cout << std::endl;

  apply_weight_update(tmp_avg_derivatives);
  ++iteration;

  update_context_errors();
}

double RafkoAutodiffGPUOptimizer::get_neuron_data(
  std::uint32_t sequence_index, std::uint32_t past_index, std::uint32_t neuron_index
){
  double ret = 0.0;
  RFASSERT(past_index < network.memory_size());
  RFASSERT_LOG(
    "Loading Neuron data from GPU Phase: sequence[{}/{}], past[{}/{}], Neuron[{}/{}], operation[{}/{}];",
    sequence_index, environment->get_number_of_sequences(),
    past_index, network.memory_size(),
    neuron_index, network.neuron_array_size(),
    get_operation_index(neuron_index), operations.size()
  );

  gpu_phase.load_output(
    &ret/*target*/, 1u/*size*/, (
      (sequence_index * network.memory_size() * operations.size())
      + ((network.memory_size() - past_index - 1) * operations.size())
      + get_operation_index(neuron_index)
    )/*offset*/
  );
  return ret;
}

} /* namespace rafko_gym */
