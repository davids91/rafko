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
  std::uint32_t asd = build_without_data(objective);
  strategy->build(operations, asd);
  gpu_phase.set_strategy(strategy);
  data.build(operations.size(), asd, environment->get_sequence_size());
  // RFASSERT_SCOPE(AUTODIFF_GPU_BUILD);
  // strategy->build(operations, build_without_data(objective));
  // gpu_phase.set_strategy(strategy);
  refresh_GPU_environment();
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
    sizeof(double) * static_cast<std::uint32_t>(network.weight_table_size())/*buffer_start_byte_offset*/,
    0u/*sequence_start_index*/, 0u/*buffer_sequence_start_index*/,
    environment->get_number_of_sequences()/*sequences_to_upload*/
  );
}

std::vector<cl::Event> RafkoAutodiffGPUOptimizer::update_labels(){
  return environment->upload_labels_to_buffer(
    opencl_queue, gpu_phase.get_input_buffer(),
    (
      sizeof(double) * static_cast<std::uint32_t>(network.weight_table_size())
      + (sizeof(double) * environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * network.input_data_size())
    )/*buffer_start_byte_offset*/,
    0u/*sequence_start_index*/, 0u/*buffer_sequence_start_index*/,
    environment->get_number_of_sequences()/*sequences_to_upload*/,
    0u/*start_index_inside_sequence*/, environment->get_sequence_size()/*sequence_truncation*/
  );
}

void RafkoAutodiffGPUOptimizer::refresh_GPU_environment(){
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
  if(refresh_environment){
    refresh_GPU_environment();
  }

  /* Reset GPU Derivatives */
  cl::Event reset_event;
  std::uint32_t output_buffer_byte_size = strategy->get_output_buffer_byte_size<double>();
  std::uint32_t weight_derivatives_byte_size = strategy->get_output_shapes().back().get_byte_size<double>();
  cl_int return_value = opencl_queue.enqueueFillBuffer<double>(
    gpu_phase.get_output_buffer(), (0.0)/* the data(pattern) value */,
    (output_buffer_byte_size - weight_derivatives_byte_size)/*offset*/,
    weight_derivatives_byte_size/*size*/,
    NULL/*events to wait for*/, &reset_event
  );
  RFASSERT( return_value == CL_SUCCESS );

  std::thread tmp_reset_thread([this](){
    std::fill(tmp_avg_derivatives.begin(), tmp_avg_derivatives.end(), 0.0);
  });

  /* Wait for derivative reset to finish before starting derivative calculation */
  return_value = reset_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  //TODO: Run for all weights
  // std::cout << "======================" << std::endl;
  gpu_phase();
  // std::cout << "======================" << std::endl;
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
  std::uint32_t i = 0;
  // std::vector<double> input_buffer_values(strategy->get_input_shapes()[0].get_number_of_elements());
  // cl_int return_value = opencl_queue.enqueueReadBuffer(
  //   gpu_phase.get_input_buffer(), CL_TRUE/*blocking*/,
  //   0U/*offset*/, (strategy->get_input_shapes()[0].get_byte_size<double>())/*size*/,
  //   input_buffer_values.data()
  // );
  // assert(return_value == CL_SUCCESS);
  // std::cout << "GPU input values:";
  // for(const double& d : input_buffer_values){
  //   if(0 == (i++ % 10))std::cout << std::endl;
  //   std::cout << "[" << d << "]";
  // }
  // std::cout << std::endl;

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

  const std::uint32_t examined_operation = 3u;
  // std::cout << "GPU output_values(op_size:" << operations.size() << "):";
  // i = 0;
  // for(const double& d : output_buffer_values){
  //   if(0 == (i % operations.size())){
  //     std::cout << std::endl << "past " << 1 - (i / operations.size()) << ":";
  //     i = 0;
  //   }
  //   if(0 == (i % 10))std::cout << "\n";
  //   if(examined_operation == (i % operations.size())) std::cout << "[ x " << d << "]";
  //     else std::cout << "[" << d << "]";
  //   ++i;
  // }
  // std::cout << std::endl;
  //
  RafkoAutodiffOptimizer::iterate();
  // std::cout << "CPU output_values:\n";
  // i = 0;
  // std::cout << "past 1: ";
  // for(const double& d : data.get_value().get_element(1u/*past_index*/)){
  //   if(0 == (i % 10))std::cout << "\n";
  //   if(examined_operation == (i % operations.size())) std::cout << "[ x " << d << "]";
  //     else std::cout << "[" << d << "]";
  //   ++i;
  // }
  // std::cout << std::endl;
  // i = 0;
  // std::cout << "past 0: ";
  // for(const double& d : data.get_value().get_element(0u/*past_index*/)){
  //   if(0 == (i % 10))std::cout << "\n";
  //   if(examined_operation == (i % operations.size())) std::cout << "[ x " << d << "]";
  //     else std::cout << "[" << d << "]";
  //   ++i;
  // }
  // std::cout << std::endl;
  // std::cout << "Weight derivatives:\n";
  // for(const double& d : tmp_avg_derivatives){
  //   std::cout << "[" << d << "]";
  // }
  // std::cout << std::endl;
  //
  std::uint32_t weight_index = 2;
  // i = 0;
  // std::cout << "CPU derivatives for weight[" << weight_index << "](past 1): \n";
  // for(std::uint32_t operation_index = 0; operation_index < operations.size(); ++operation_index){
  //   if(0 == (i++ % 10))std::cout << "\n";
  //   std::cout << "[" << data.get_derivative(1/*past_index*/, operation_index, weight_index) << "]";
  // }
  // std::cout << std::endl;
  // i = 0;
  // std::cout << "CPU derivatives for weight[" << weight_index << "](past 0): \n";
  // for(std::uint32_t operation_index = 0; operation_index < operations.size(); ++operation_index){
  //   if(0 == (i++ % 10))std::cout << "\n";
  //   std::cout << "[" << data.get_derivative(0/*past_index*/, operation_index, weight_index) << "]";
  // }
  // std::cout << std::endl;
  // i = 0;
  // std::cout << "CPU derivatives for operation[" << examined_operation << "](past 0): \n";
  // for(std::uint32_t d_w_index = 0; d_w_index < network.weight_table_size(); ++d_w_index){
  //   if(0 == (i++ % 10))std::cout << "(" << i << ")\n";
  //   std::cout << "[" << data.get_derivative(0/*past_index*/, 0/*operation_index*/, d_w_index) << "]";
  // }
  // std::cout << std::endl;
  apply_weight_update(tmp_avg_derivatives);
  ++iteration;
  update_context_errors();
  // RFASSERT(0);
}

double RafkoAutodiffGPUOptimizer::get_neuron_data(
  std::uint32_t sequence_index, std::uint32_t past_index, std::uint32_t neuron_index
){
  double ret = 0.0;
  RFASSERT(past_index < network.memory_size());
  RFASSERT_LOG(
    "Loading Neuron data from GPU Phase: sequence[{}/{}], past[{}/{}], Neuron[{}/{}], operation[{}/{}] ==> offset: {}",
    sequence_index, environment->get_number_of_sequences(),
    past_index, network.memory_size(),
    neuron_index, network.neuron_array_size(),
    get_operation_index(neuron_index), operations.size(),
    (
      (sequence_index * network.memory_size() * operations.size())
      + ((network.memory_size() - past_index - 1) * operations.size())
      + get_operation_index(neuron_index)
    )
  );
  // sequence[0/1], past[1/2], Neuron[3/4], operation[1/37] ==> offset: 38
  // sequence[0/1], past[0/2], Neuron[3/4], operation[1/37] ==> offset: 75

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
