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

  /* Reset GPU Derivatives and triggered derivative operation count */
  cl::Event reset_event;
  std::uint32_t output_buffer_byte_size = strategy->get_output_buffer_byte_size<double>();
  std::uint32_t weight_derivatives_byte_size = strategy->get_output_shapes().back().get_byte_size<double>();
  cl_int return_value = opencl_queue.enqueueFillBuffer<double>(
    gpu_phase.get_output_buffer(), (0.0)/* the data(pattern) value */,
    (output_buffer_byte_size - weight_derivatives_byte_size)/*offset*/,
    (weight_derivatives_byte_size)/*size*/,
    NULL/*events to wait for*/, &reset_event
  );
  RFASSERT( return_value == CL_SUCCESS );

  cl::Event sequence_start_index_event;
  return_value = opencl_queue.enqueueFillBuffer<double>(
    gpu_phase.get_input_buffer(), static_cast<double>(rand()%(std::max(1,
      static_cast<std::int32_t>(environment->get_number_of_sequences())
      - static_cast<std::int32_t>(settings.get_minibatch_size())
    )))/*the data(pattern) value*/,
    (strategy->get_input_buffer_byte_size<double>() - (sizeof(double) * 3)),/*offset*/
    sizeof(double)/*size*/, NULL/*events to wait for*/, &sequence_start_index_event
  );
  RFASSERT( return_value == CL_SUCCESS );

  cl::Event truncation_event;
  return_value = opencl_queue.enqueueFillBuffer<double>(
    gpu_phase.get_input_buffer(), static_cast<double>(settings.get_memory_truncation())/*the data(pattern) value*/,
    (strategy->get_input_buffer_byte_size<double>() - (sizeof(double) * 2)),/*offset*/
    sizeof(double)/*size*/, NULL/*events to wait for*/, &truncation_event
  );
  RFASSERT( return_value == CL_SUCCESS );

  /* Wait for preparing operations to finish before starting derivative calculation */
  return_value = reset_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  return_value = truncation_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  std::thread tmp_reset_thread([this](){
    std::fill(tmp_avg_d.begin(), tmp_avg_d.end(), 0.0);
  });

  for(std::int32_t d_w_index = 0; d_w_index < network.weight_table_size(); ++d_w_index){
    cl::Event d_w_index_event;
    return_value = opencl_queue.enqueueFillBuffer<double>(
      gpu_phase.get_input_buffer(), static_cast<double>(d_w_index)/*the data(pattern) value*/,
      (strategy->get_input_buffer_byte_size<double>() - sizeof(double))/*offset*/,
      sizeof(double)/*size*/, NULL/*events to wait for*/, &d_w_index_event
    );
    RFASSERT( return_value == CL_SUCCESS );
    return_value = d_w_index_event.wait();
    RFASSERT( return_value == CL_SUCCESS );

    gpu_phase();
  }

  tmp_reset_thread.join();
  RFASSERT_LOG("sequence count: {}", environment->get_number_of_sequences());
  RFASSERT_LOG("inputs in one sequence: {}", environment->get_inputs_in_one_sequence());
  RFASSERT_LOG("operations count: {}", operations.size());
  RFASSERT_LOG("weights count: {}", network.weight_table_size());
  RFASSERT_LOG(
    "Getting Autodiff Phase weight derivatives({} numbers) from: [{}]",
    tmp_avg_d.size(),
    ( /* operation values + operation derivatives size */
      (environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * operations.size())
      + (environment->get_number_of_sequences() * environment->get_sequence_size() * operations.size())
    )/*offset*/
  );
  gpu_phase.load_output(
    tmp_avg_d.data()/*target*/, tmp_avg_d.size()/*size*/,
    ( /* operation values + operation derivatives size */
      (environment->get_number_of_sequences() * environment->get_inputs_in_one_sequence() * operations.size())
      + (environment->get_number_of_sequences() * environment->get_sequence_size() * operations.size())
    )/*offset*/
  );

  RFASSERT( static_cast<std::int32_t>(tmp_avg_d.size()) > std::count(tmp_avg_d.begin(),tmp_avg_d.end(), 0.0));
  apply_weight_update(tmp_avg_d);

  ++iteration;
  update_context_errors();
}

double RafkoAutodiffGPUOptimizer::get_neuron_data(
  std::uint32_t sequence_index, std::uint32_t past_index, std::uint32_t neuron_index
){
  double ret = 0.0;
  RFASSERT(past_index < network.memory_size());
  RFASSERT_LOG(
    "Loading Neuron data from GPU Phase: sequence[{}/{}], past[{}], Neuron[{}/{}], operation[{}/{}] ==> offset: {}",
    sequence_index, environment->get_number_of_sequences(),
    past_index, neuron_index, network.neuron_array_size(),
    get_operation_index(neuron_index), operations.size(),
    (
      (sequence_index * environment->get_inputs_in_one_sequence() * operations.size())
      + ((environment->get_inputs_in_one_sequence() - 1 - past_index) * operations.size())
      + get_operation_index(neuron_index)
    )
  );

  gpu_phase.load_output(
    &ret/*target*/, 1u/*size*/, (
      (sequence_index * environment->get_inputs_in_one_sequence() * operations.size())
      + ((environment->get_inputs_in_one_sequence() - 1 - past_index) * operations.size())
      + get_operation_index(neuron_index)
    )/*offset*/
  );
  return ret;
}

double RafkoAutodiffGPUOptimizer::get_avg_gradient(std::uint32_t d_w_index) const{
  RFASSERT(static_cast<std::int32_t>(d_w_index) < network.weight_table_size());
  double d_w_index_gradient;
  gpu_phase.load_output(
    &d_w_index_gradient/*target*/, 1/*size*/,
    ( /* End of the buffer - number of weights + weight_index */
      strategy->get_output_shapes().back().get_number_of_elements()
       - network.weight_table_size() + d_w_index
    )/*offset*/
  );
  return d_w_index_gradient;
}

} /* namespace rafko_gym */
