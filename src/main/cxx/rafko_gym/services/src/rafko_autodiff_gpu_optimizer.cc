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
#include "rafko_gym/services/rafko_autodiff_gpu_optimizer.hpp"

namespace rafko_gym {

void RafkoAutodiffGPUOptimizer::build(
    std::shared_ptr<RafkoDataSet> data_set,
    std::shared_ptr<RafkoObjective> objective) {
  RFASSERT_SCOPE(AUTODIFF_GPU_BUILD);
  m_usedSequenceTruncation = std::min(m_settings->get_memory_truncation(),
                                      data_set->get_sequence_size());
  m_usedMinibatchSize = std::min(m_settings->get_minibatch_size(),
                                 data_set->get_number_of_sequences());
  if (m_trainingEvaluator)
    m_trainingEvaluator->set_data_set(data_set);

  const std::uint32_t derivative_relevant_operation_count =
      build_without_data(data_set, objective);

  AutoDiffGPUStrategy strategy(m_openclDevice, *m_settings, m_network,
                               m_neuronIndexToSpikeOperationIndex, data_set);
  strategy.build(m_operations, derivative_relevant_operation_count);

  m_gpuPhase.set_strategy(
      std::make_shared<AutoDiffGPUStrategy>(std::move(strategy)));
  sync_data_set_on_GPU(*data_set);
  upload_network();
  m_built = true;
}

void RafkoAutodiffGPUOptimizer::upload_weight_table() {
  RFASSERT_LOG("Uploading network weight table(size: {} bytes) to device..",
               (sizeof(double) * m_network.weight_table_size()));
  cl_int return_value = m_openclQueue.enqueueWriteBuffer(
      m_gpuPhase.get_input_buffer(), CL_TRUE /*blocking*/, 0u /*offset*/,
      (sizeof(double) * m_network.weight_table_size()) /*size*/,
      m_network.mutable_weight_table()->Mutable(0));
  RFASSERT(return_value == CL_SUCCESS);
}

void RafkoAutodiffGPUOptimizer::upload_network() {
  /*!Note: byte_ofset is aligned to sizeof(double), because it is based on the
   *previous entries in front of it in the input buffer. Since
   *enqueueWriteBuffer acepts byte size it is fine to use sizeof(std::uint32_t)
   *for the size. Only the offset needs to be aligned, because accessing it
   *inside the kernel is with the help of the strategy @get_input_shapes sizes.
   * The only requirement here is that propagation_instructions_byte_size needs
   *to be bigger or equal to the actual data to be accessed
   **/
  const std::vector<std::uint32_t> &propagation_instructions =
      static_cast<const AutoDiffGPUStrategy &>(m_gpuPhase.expose_strategy())
          .get_propagation_instructions();

  const std::uint32_t weight_table_byte_size =
      sizeof(double) * m_gpuPhase.expose_strategy().get_input_shapes()[0][0];
  const std::uint32_t inputs_byte_size =
      sizeof(double) * m_gpuPhase.expose_strategy().get_input_shapes()[0][1];
  const std::uint32_t labels_byte_size =
      sizeof(double) * m_gpuPhase.expose_strategy().get_input_shapes()[0][2];

  const std::uint32_t neuron_map_byte_offset =
      (weight_table_byte_size + inputs_byte_size + labels_byte_size);
  const std::uint32_t neuron_map_byte_size =
      (sizeof(std::uint32_t) * m_neuronIndexToSpikeOperationIndex.size());
  RFASSERT_LOG("Uploading neuron to spike operation map(offset: {}, byte size: "
               "{} bytes) to device..",
               neuron_map_byte_offset, neuron_map_byte_size);
  [[maybe_unused]] cl_int return_value = m_openclQueue.enqueueWriteBuffer(
      m_gpuPhase.get_input_buffer(), CL_TRUE /*blocking*/,
      neuron_map_byte_offset, neuron_map_byte_size,
      m_neuronIndexToSpikeOperationIndex.data());
  RFASSERT(return_value == CL_SUCCESS);

  const std::uint32_t neuron_map_size =
      sizeof(double) * m_gpuPhase.expose_strategy().get_input_shapes()[0][3];
  const std::uint32_t propagation_blob_byte_offset =
      (weight_table_byte_size + inputs_byte_size + labels_byte_size +
       neuron_map_size);
  const std::uint32_t propagation_instructions_byte_size =
      (sizeof(std::uint32_t) * propagation_instructions.size());

  RFASSERT_LOG("Uploading network propagation table(offset: {}, size: {} "
               "bytes) to device..",
               propagation_blob_byte_offset, propagation_instructions.size());
  return_value = m_openclQueue.enqueueWriteBuffer(
      m_gpuPhase.get_input_buffer(), CL_TRUE /*blocking*/,
      propagation_blob_byte_offset, propagation_instructions_byte_size,
      propagation_instructions.data());
  RFASSERT(return_value == CL_SUCCESS);
}

std::vector<cl::Event>
RafkoAutodiffGPUOptimizer::update_inputs(const RafkoDataSet &data_set) {
  return data_set.upload_inputs_to_buffer(
      m_openclQueue, m_gpuPhase.get_input_buffer(),
      sizeof(double) *
          static_cast<std::uint32_t>(
              m_network.weight_table_size()) /*buffer_start_byte_offset*/,
      0u /*sequence_start_index*/, 0u /*buffer_sequence_start_index*/,
      data_set.get_number_of_sequences() /*sequences_to_upload*/
  );
}

std::vector<cl::Event>
RafkoAutodiffGPUOptimizer::update_labels(const RafkoDataSet &data_set) {
  return data_set.upload_labels_to_buffer(
      m_openclQueue, m_gpuPhase.get_input_buffer(),
      (sizeof(double) *
           static_cast<std::uint32_t>(m_network.weight_table_size()) +
       (sizeof(double) * data_set.get_number_of_sequences() *
        data_set.get_inputs_in_one_sequence() *
        m_network.input_data_size())) /*buffer_start_byte_offset*/,
      0u /*sequence_start_index*/, 0u /*buffer_sequence_start_index*/,
      data_set.get_number_of_sequences() /*sequences_to_upload*/,
      0u /*start_index_inside_sequence*/,
      data_set.get_sequence_size() /*sequence_truncation*/
  );
}

void RafkoAutodiffGPUOptimizer::sync_data_set_on_GPU(
    const RafkoDataSet &data_set) {
  std::vector<cl::Event> input_events = update_inputs(data_set);
  std::vector<cl::Event> label_events = update_labels(data_set);
  for (cl::Event &e : input_events) {
    cl_int return_value = e.wait();
    RFASSERT(return_value == CL_SUCCESS);
  }
  for (cl::Event &e : label_events) {
    cl_int return_value = e.wait();
    RFASSERT(return_value == CL_SUCCESS);
  }
}

void RafkoAutodiffGPUOptimizer::iterate(const RafkoDataSet &data_set,
                                        bool force_gpu_upload) {
  RFASSERT_SCOPE(AUTODIFF_GPU_ITERATE);
  RFASSERT(data_set.get_feature_size() == m_network.output_neuron_number());

  upload_weight_table();
  if (force_gpu_upload) {
    sync_data_set_on_GPU(data_set);
  }

  cl::Event sequence_start_index_event;
  const std::uint32_t sequences_start =
      rand() %
      (std::max(
          1, static_cast<std::int32_t>(data_set.get_number_of_sequences()) -
                 static_cast<std::int32_t>(m_settings->get_minibatch_size())));
  [[maybe_unused]] cl_int return_value =
      m_openclQueue.enqueueFillBuffer<double>(
          m_gpuPhase.get_input_buffer(),
          static_cast<double>(sequences_start) /*the data(pattern) value*/,
          (m_gpuPhase.expose_strategy().get_input_buffer_byte_size<double>() -
           (sizeof(double) * 2)), /*offset*/
          sizeof(double) /*size*/, NULL /*events to wait for*/,
          &sequence_start_index_event);
  RFASSERT(return_value == CL_SUCCESS);

  cl::Event truncation_event;
  return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_gpuPhase.get_input_buffer(),
      static_cast<double>(
          m_settings->get_memory_truncation()) /*the data(pattern) value*/,
      (m_gpuPhase.expose_strategy().get_input_buffer_byte_size<double>() -
       (sizeof(double) * 1)), /*offset*/
      sizeof(double) /*size*/, NULL /*events to wait for*/, &truncation_event);
  RFASSERT(return_value == CL_SUCCESS);

  /* Wait for preparing operations to finish before starting derivative
   * calculation */
  return_value = truncation_event.wait();
  RFASSERT(return_value == CL_SUCCESS);

  std::thread tmp_reset_thread(
      [this]() { std::fill(m_tmpAvgD.begin(), m_tmpAvgD.end(), 0.0); });
  m_gpuPhase();

  RFASSERT_LOG("sequence count: {}", data_set.get_number_of_sequences());
  RFASSERT_LOG("inputs in one sequence: {}",
               data_set.get_inputs_in_one_sequence());
  RFASSERT_LOG("operations count: {}", m_operations.size());
  RFASSERT_LOG("weights count: {}", m_network.weight_table_size());
  RFASSERT_LOG(
      "Getting Autodiff Phase weight derivatives({} numbers) from: [{}]",
      m_tmpAvgD.size(),
      (/* operation values + operation derivatives size */
       (data_set.get_number_of_sequences() *
        data_set.get_inputs_in_one_sequence() * m_operations.size()) +
       (data_set.get_number_of_sequences() * data_set.get_sequence_size() *
        m_operations.size())) /*offset*/
  );
  tmp_reset_thread.join();
  auto output_shape = m_gpuPhase.expose_strategy().get_output_shapes().back();
  m_gpuPhase.load_output(m_tmpAvgD.data() /*target*/, m_tmpAvgD.size() /*size*/,
                         /* operation values + operation derivatives size */
                         (output_shape[0] + output_shape[1]) /*offset*/
  );

  if (static_cast<std::int32_t>(m_tmpAvgD.size()) >
      std::count(m_tmpAvgD.begin(), m_tmpAvgD.end(), 0.0)) {
    apply_weight_update(m_tmpAvgD);
  }

  ++m_iteration;
  update_context_errors(force_gpu_upload);
}

double RafkoAutodiffGPUOptimizer::get_neuron_data(
    std::uint32_t sequence_index, std::uint32_t past_index,
    std::uint32_t neuron_index, const RafkoDataSet &data_set) {
  double ret = 0.0;
  RFASSERT(past_index <= m_network.memory_size());
  RFASSERT_LOG("Loading Neuron data from GPU Phase: sequence[{}/{}], past[{}], "
               "Neuron[{}/{}], operation[{}/{}] ==> offset: {}",
               sequence_index, data_set.get_number_of_sequences(), past_index,
               neuron_index, m_network.neuron_array_size(),
               get_operation_index(neuron_index), m_operations.size(),
               ((sequence_index * data_set.get_inputs_in_one_sequence() *
                 m_operations.size()) +
                ((data_set.get_inputs_in_one_sequence() - 1 - past_index) *
                 m_operations.size()) +
                get_operation_index(neuron_index)));

  m_gpuPhase.load_output(
      &ret /*target*/, 1u /*size*/,
      ((sequence_index * data_set.get_inputs_in_one_sequence() *
        m_operations.size()) +
       ((data_set.get_inputs_in_one_sequence() - 1 - past_index) *
        m_operations.size()) +
       get_operation_index(neuron_index)) /*offset*/
  );
  return ret;
}

double
RafkoAutodiffGPUOptimizer::get_avg_gradient(std::uint32_t d_w_index) const {
  RFASSERT(static_cast<std::int32_t>(d_w_index) <
           m_network.weight_table_size());
  double d_w_index_gradient;
  m_gpuPhase.load_output(
      &d_w_index_gradient /*target*/, 1 /*size*/,
      (/* End of the buffer - number of weights + weight_index */
       m_gpuPhase.expose_strategy()
           .get_output_shapes()
           .back()
           .get_number_of_elements() -
       m_network.weight_table_size() + d_w_index) /*offset*/
  );
  return d_w_index_gradient;
}

} /* namespace rafko_gym */
