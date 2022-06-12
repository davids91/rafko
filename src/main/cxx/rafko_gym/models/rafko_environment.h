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

#ifndef RAFKO_ENVIRONMENT_H
#define RAFKO_ENVIRONMENT_H

#include "rafko_global.h"

#if(RAFKO_USES_OPENCL)
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_agent.h"

namespace RAFKO_FULL_EXPORT rafko_gym {

/**
 * @brief      A class representing an environment, producing fitness/error value. Error values are negative, while fittness
 *             values are positive
 */
class RAFKO_FULL_EXPORT RafkoEnvironment{
public:

  /**
   * @brief      Saves the RafkoEnvironment state
   */
  virtual void push_state() = 0;

  /**
   * @brief      Restores the previously stored environment state
   */
  virtual void pop_state() = 0;

  /**
   * @brief      Gets an input sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The input sample.
   */
  virtual const std::vector<double>& get_input_sample(std::uint32_t raw_input_index) const = 0;

  /**
   * @brief      Gets an input sample from the set
   *
   * @return     A const reference of the input sample.
   */
  virtual const std::vector<std::vector<double>>& get_input_samples() const = 0;

  /**
   * @brief      Gets a label sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The label sample.
   */
  virtual const std::vector<double>& get_label_sample(std::uint32_t raw_label_index) const = 0;

  /**
   * @brief      Gets a label sample from the set
   *
   * @return     A const reference of the label samples array
   */
  virtual const std::vector<std::vector<double>>& get_label_samples() const = 0;

  /**
   * @brief      Gets the number of floating point values the evaluation accepts to produce the label values
   *
   * @return     The feature size.
   */
  virtual std::uint32_t get_input_size() const = 0;

  /**
   * @brief      Gets the number of values present in the output
   *
   * @return     The feature size.
   */
  virtual std::uint32_t get_feature_size() const = 0;

  /**
   * @brief      Gets the number of raw input arrays stored in the pbject
   *
   * @return     The number of input samples.
   */
  virtual std::uint32_t get_number_of_input_samples() const = 0;

  /**
   * @brief      The number of raw label arrays stored in the object
   *
   * @return     The number of labels.
   */
  virtual std::uint32_t get_number_of_label_samples() const = 0;

  /**
   * @brief      Gets the number of sequences stored in the object. One sequence contains
   *             a number of input and label sample arrays. There might be more input arrays,
   *             than label arrays in one sequences. The difference is given by @get_prefill_inputs_number
   *
   * @return     The number of sequences.
   */
  virtual std::uint32_t get_number_of_sequences() const = 0;

  /**
   * @brief      Gets the size of one sequence
   *
   * @return     Number of consecutive datapoints that count as one sample.
   */
  virtual std::uint32_t get_sequence_size() const = 0;

  /**
   * @brief      Gets the number of inputs to be used as initializing the network during a training run
   *
   * @return     The number of inputs to be used for network initialization during training
   */
  virtual std::uint32_t get_prefill_inputs_number() const = 0;

  virtual ~RafkoEnvironment() = default;

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Upload inputs to the provided buffer
   *
   * @param       opencl_queue                  The OpenCL queue to start the buffer oprations on
   * @param       buffer                        The buffer to upload the information to
   * @param       buffer_start_byte_offset      The offset pointing to the beginning of the area the sequences are uploaded to
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  std::vector<cl::Event> upload_inputs_to_buffer(
    cl::CommandQueue opencl_queue, cl::Buffer buffer, std::uint32_t buffer_start_byte_offset,
    std::uint32_t sequence_start_index, std::uint32_t buffer_sequence_start_index,
    std::uint32_t sequences_to_upload
  );

  /**
   * @brief     Upload labels to the error phase to be able to evaluate agent output
   *
   * @param       opencl_queue                  The OpenCL queue to start the buffer oprations on
   * @param       buffer                        The buffer to upload the information to
   * @param       buffer_start_byte_offset      The offset pointing to the beginning of the area the sequences are uploaded to
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   * @param[in]   start_index_inside_sequence   Start index inside sequence for sequence truncation
   * @param[in]   sequence_truncation           Number of labels to evaluate per sequence (sequence truncation size)
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  std::vector<cl::Event> upload_labels_to_buffer(
    cl::CommandQueue opencl_queue, cl::Buffer buffer, std::uint32_t buffer_start_byte_offset,
    std::uint32_t sequence_start_index, std::uint32_t buffer_sequence_start_index,
    std::uint32_t sequences_to_upload, std::uint32_t start_index_inside_sequence,
    std::uint32_t sequence_truncation
  );
  #endif/*(RAFKO_USES_OPENCL)*/

};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */
