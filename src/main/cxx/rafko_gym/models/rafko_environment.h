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

#include "rafko_gym/models/rafko_agent.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      A class representing an environment, producing fitness/error value. Error values are negative, while fittness
 *             values are positive
 */
class RafkoEnvironment{
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
  virtual const std::vector<sdouble32>& get_input_sample(uint32 raw_input_index)const = 0;

  /**
   * @brief      Gets an input sample from the set
   *
   * @return     A const reference of the input sample.
   */
  virtual const std::vector<std::vector<sdouble32>>& get_input_samples()const = 0;

  /**
   * @brief      Gets a label sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The label sample.
   */
  virtual const std::vector<sdouble32>& get_label_sample(uint32 raw_label_index)const = 0;

  /**
   * @brief      Gets a label sample from the set
   *
   * @return     A const reference of the label samples array
   */
  virtual const std::vector<std::vector<sdouble32>>& get_label_samples()const = 0;

  /**
   * @brief      Gets the number of floating point values the evaluation accepts to produce the label values
   *
   * @return     The feature size.
   */
  virtual uint32 get_input_size()const = 0;

  /**
   * @brief      Gets the number of values present in the output
   *
   * @return     The feature size.
   */
  virtual uint32 get_feature_size()const = 0;

  /**
   * @brief      Gets the number of raw input arrays stored in the pbject
   *
   * @return     The number of input samples.
   */
  virtual uint32 get_number_of_input_samples()const = 0;

  /**
   * @brief      The number of raw label arrays stored in the object
   *
   * @return     The number of labels.
   */
  virtual uint32 get_number_of_label_samples()const = 0;

  /**
   * @brief      Gets the number of sequences stored in the object. One sequence contains
   *             a number of input and label sample arrays. There might be more input arrays,
   *             than label arrays in one sequences. The difference is given by @get_prefill_inputs_number
   *
   * @return     The number of sequences.
   */
  virtual uint32 get_number_of_sequences()const = 0;

  /**
   * @brief      Gets the size of one sequence
   *
   * @return     Number of consecutive datapoints that count as one sample.
   */
  virtual uint32 get_sequence_size()const = 0;

  /**
   * @brief      Gets the number of inputs to be used as initializing the network during a training run
   *
   * @return     The number of inputs to be used for network initialization during training
   */
  virtual uint32 get_prefill_inputs_number()const = 0;

  virtual ~RafkoEnvironment() = default;
};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */
