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

#ifndef RAFKO_OBJECTIVE_H
#define RAFKO_OBJECTIVE_H

#include "rafko_global.h"

#include <vector>

namespace rafko_gym{

/**
 * @brief      This class
 */
class RAFKO_FULL_EXPORT RafkoObjective{
public:

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   */
  virtual void set_features_for_sequences(
    const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  ) = 0;

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the dataset labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   */
  virtual void set_features_for_sequences(
    const std::vector<std::vector<sdouble32>>& neuron_data, uint32 neuron_buffer_index,
    uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation,
    std::vector<sdouble32>& tmp_data
  ) = 0;

  /**
   * @brief      Provides the last measured overall fitness value
   *
   * @return     A vector of error/fitness value of all the installed feature arrays
   */
  virtual const std::vector<sdouble32>& get_feature_fitness_vector()const = 0;

  /**
   * @brief      Provides the last measured overall fitness value
   *
   * @return     A vector of error/fitness value of all the installed feature arrays
   */
  virtual sdouble32 get_feature_fitness()const = 0;

  /**
   * @brief      Sets the error values to the default value
   */
  virtual void reset_errors() = 0;

  /**
   * @brief      Saves the RafkoEnvironment state
   */
  virtual void push_state() = 0;

  /**
   * @brief      Restores the previously stored environment state
   */
  virtual void pop_state() = 0;

  /**
   * @brief     Puts the set in a thread-safe state, enabling multi-threaded set access to the error_values vector, but
   *            disabling error_sum calculations(one of the main common part of the set).
   */
  virtual void expose_to_multithreading() = 0;

  /**
   * @brief     Restores the set to a non-thread-safe state, disabling multi-threaded set access to the error_values vector, but
   *            re-enabling error_sum calculations(one of the main common part of the set). Also re-calculates error value sum
   */
  virtual void conceal_from_multithreading() = 0;

  virtual ~RafkoObjective() = default;
};

} /* namespace rafko_gym */
#endif /* RAFKO_OBJECTIVE_H */
