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
#include "rafko_gym/models/rafko_dataset.hpp"

#include <iostream> //TODO: Remove this debug line

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym {

#if (RAFKO_USES_OPENCL)
std::vector<cl::Event> RafkoDataSet::upload_inputs_to_buffer(
    cl::CommandQueue opencl_queue, cl::Buffer buffer,
    std::uint32_t buffer_start_byte_offset, std::uint32_t sequence_start_index,
    std::uint32_t buffer_sequence_start_index,
    std::uint32_t sequences_to_upload) const {
  [[maybe_unused]] cl_int return_value;
  RFASSERT_LOG("Uploading agent inputs: sequence start index: {}, sequence "
               "start index in buffer: {}, sequences to upload: {}",
               sequence_start_index, buffer_sequence_start_index,
               sequences_to_upload);
  std::uint32_t elements_in_a_sequence =
      get_sequence_size() + get_prefill_inputs_number();
  /*!Note: elements == inputs */
  std::uint32_t raw_input_start =
      (sequence_start_index * elements_in_a_sequence);
  std::uint32_t raw_input_num = (sequences_to_upload * elements_in_a_sequence);
  std::uint32_t input_buffer_byte_offset =
      (buffer_start_byte_offset +
       (buffer_sequence_start_index * elements_in_a_sequence *
        get_input_size() * sizeof(double)));
  RFASSERT_LOG("starting offset: {}; input size: {}; sequence size: {}; "
               "Resulting offset: {}",
               buffer_start_byte_offset, get_input_size(), get_sequence_size(),
               input_buffer_byte_offset);
  std::vector<cl::Event> events(raw_input_num);

  RFASSERT((raw_input_start + raw_input_num) <= get_number_of_input_samples());
  for (std::uint32_t raw_input_index = raw_input_start;
       raw_input_index < (raw_input_start + raw_input_num); ++raw_input_index) {
    RFASSERT_LOG("Input buffer byte offset: {}", input_buffer_byte_offset);
    return_value = opencl_queue.enqueueWriteBuffer(
        buffer, CL_FALSE /*blocking*/, input_buffer_byte_offset /*offset*/,
        (sizeof(double) * get_input_sample(raw_input_index).size()) /*size*/,
        get_input_sample(raw_input_index).data(), NULL,
        &events[raw_input_index - raw_input_start]);
    RFASSERT(return_value == CL_SUCCESS);
    input_buffer_byte_offset +=
        (sizeof(double) * get_input_sample(raw_input_index).size());
  }
  return events;
}

std::vector<cl::Event> RafkoDataSet::upload_labels_to_buffer(
    cl::CommandQueue opencl_queue, cl::Buffer buffer,
    std::uint32_t buffer_start_byte_offset, std::uint32_t sequence_start_index,
    std::uint32_t buffer_sequence_start_index,
    std::uint32_t sequences_to_upload,
    std::uint32_t start_index_inside_sequence,
    std::uint32_t sequence_truncation) const {
  [[maybe_unused]] cl_int return_value;
  RFASSERT_LOG(
      "Uploading labels to evaluate: sequence start index: {}, sequence start "
      "index in buffer: {}, buffer labels byte offset: {} sequences to upload: "
      "{}; start index inside sequence: {}; sequence truncation: {}",
      sequence_start_index, buffer_sequence_start_index,
      buffer_start_byte_offset, sequences_to_upload,
      start_index_inside_sequence, sequence_truncation);
  std::uint32_t elements_in_a_sequence = get_sequence_size();
  /*!Note: elements == labels */
  std::uint32_t raw_label_start =
      (sequence_start_index * elements_in_a_sequence);
  std::uint32_t raw_label_num = (sequences_to_upload * elements_in_a_sequence);

  RFASSERT((raw_label_start + raw_label_num) <= get_number_of_label_samples());
  RFASSERT((start_index_inside_sequence + sequence_truncation) <=
           get_sequence_size());
  RFASSERT(0u < sequence_truncation);

  const std::uint32_t buffer_byte_offset =
      (buffer_start_byte_offset +
       (buffer_sequence_start_index * sequence_truncation * get_feature_size() *
        sizeof(double)));
  RFASSERT_LOG("starting offset: {}; feature size: {}; sequence size: {}; "
               "Resulting offset: {}",
               buffer_start_byte_offset, get_feature_size(),
               get_sequence_size(), buffer_byte_offset);

  std::uint32_t labels_byte_offset = 0u;
  const std::uint32_t label_byte_size = (sizeof(double) * get_feature_size());
  std::vector<cl::Event> events(sequences_to_upload * sequence_truncation);
  for (std::uint32_t sequence_index = sequence_start_index;
       sequence_index < (sequence_start_index + sequences_to_upload);
       ++sequence_index) {
    std::uint32_t truncated_start =
        (sequence_index * elements_in_a_sequence) + start_index_inside_sequence;
    std::uint32_t uploaded_label_index =
        (sequence_index - sequence_start_index) * sequence_truncation;
    for (std::uint32_t truncated_index = truncated_start;
         truncated_index < (truncated_start + sequence_truncation);
         ++truncated_index) {
      RFASSERT_LOG("used offset for label[{}]: {} ( + {})", truncated_index,
                   (buffer_byte_offset + labels_byte_offset), label_byte_size);
      // std::cout << "label sample during upload: ";
      // for(double d : get_label_sample(truncated_index))std::cout << "[" << d
      // << "]"; std::cout << std::endl; std::cout << "get_feature_size():" <<
      // get_feature_size() << std::endl; std::cout << "label_byte_size:" <<
      // label_byte_size << std::endl; std::cout << "buffer_byte_offset:" <<
      // buffer_byte_offset << std::endl; std::cout << "labels_byte_offset:" <<
      // labels_byte_offset << std::endl; std::cout << "truncated_index:" <<
      // truncated_index << std::endl; std::cout << "====\n";
      return_value = opencl_queue.enqueueWriteBuffer(
          buffer, CL_FALSE /*blocking*/,
          (buffer_byte_offset + labels_byte_offset) /*offset*/,
          label_byte_size /*size*/, get_label_sample(truncated_index).data(),
          NULL,
          &events[uploaded_label_index + truncated_index - truncated_start]);
      RFASSERT(return_value == CL_SUCCESS);
      labels_byte_offset += label_byte_size;
    }
  }
  // std::cout << "============" << std::endl;

  return events;
}
#endif /*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */
