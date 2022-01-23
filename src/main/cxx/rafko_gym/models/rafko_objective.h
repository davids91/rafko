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

#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_gym/models/rafko_environment.h"

namespace rafko_gym{

/**
 * @brief      This class
 */
class RAFKO_FULL_EXPORT RafkoObjective
#if(RAFKO_USES_OPENCL)
: public rafko_mainframe::RafkoGPUStrategyPhase
#endif/*(RAFKO_USES_OPENCL)*/
{
public:

  /**
   * @brief      Sets the approximated value for an observed value and provides the calculated fitness.
   *             assumes that the sequence size of the @environment is 1
   *
   * @param[in]  environment                  The environment to evaluate the provided Neuron data on
   * @param[in]  sample_index             The sample index inside the environment
   * @param[in]  neuron_data              The neuron data to evaluate
   * @return     The resulting error
   */
  virtual sdouble32 set_feature_for_label(const rafko_gym::RafkoEnvironment& environment, uint32 sample_index, const std::vector<sdouble32>& neuron_data) = 0;

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  environment                  The environment to evaluate the provided Neuron data on
   * @param[in]  neuron_data              The neuron data
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  raw_start_index          The raw start index inside the environment labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  labels_to_evaluate       The labels to evaluate
   * @return     The resulting error
   */
  virtual sdouble32 set_features_for_labels(
     const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 raw_start_index, uint32 labels_to_evaluate
  ) = 0;

  /**
   * @brief      Provides the fitness value
   *
   * @param[in]  environment                  The environment to evaluate the provided Neuron data on
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the environment labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   * @return     The resulting error
   */
  virtual sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  ) = 0;

  /**
   * @brief      Same as @set_feature_for_label but in bulk
   *
   * @param[in]  environment                  The environment to evaluate the provided Neuron data on
   * @param[in]  neuron_data              The neuron data containing every output data for the @sequences_to_evaluate
   * @param[in]  neuron_buffer_index      The index of the outer neuron bufer to start evaluation from
   * @param[in]  sequence_start_index     The raw start index inside the environment labels; Meaning the index inside the labels array, which contains the samples(each with possible multiple labels in sequential order)
   * @param[in]  sequences_to_evaluate    The labels to evaluate
   * @param[in]  start_index_in_sequence  The starting index inside each sequence to update the labels
   * @param[in]  sequence_truncation      The sequence truncation
   * @return     The resulting error
   */
  virtual sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation, std::vector<sdouble32>& tmp_data
  ) = 0;

  #if(RAFKO_USES_OPENCL)
  virtual void set_gpu_parameters(uint32 pairs_to_evaluate_, uint32 feature_size_) = 0;
  #endif/*(RAFKO_USES_OPENCL)*/


  virtual ~RafkoObjective() = default;
};

} /* namespace rafko_gym */
#endif /* RAFKO_OBJECTIVE_H */
