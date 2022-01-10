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
 *    along with Foobar.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef RAFKO_DATASET_COST_H
#define RAFKO_DATASET_COST_H

#include "rafko_global.h"

#include <vector>
#include <memory>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_utilities/models/data_pool.h"
#include "rafko_gym/services/cost_function.h"
#include "rafko_gym/services/function_factory.h"

#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_agent.h"

namespace rafko_gym{

/**
 * @brief      Error Statistics for a @RafkoDatasetWrapper
 */
class RAFKO_FULL_EXPORT RafkoDatasetCost : public RafkoObjective{
public:
  RafkoDatasetCost(rafko_mainframe::RafkoSettings& settings_, std::shared_ptr<rafko_gym::CostFunction> cost_function_)
  : settings(settings_)
  , cost_function(cost_function_)
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  RafkoDatasetCost(rafko_mainframe::RafkoSettings& settings_, rafko_gym::Cost_functions the_function)
  : settings(settings_)
  , cost_function(rafko_gym::FunctionFactory::build_cost_function(the_function, settings_))
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  sdouble32 set_feature_for_label(const rafko_gym::RafkoEnvironment& environment, uint32 sample_index, const std::vector<sdouble32>& neuron_data);

  sdouble32 set_features_for_labels(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 raw_start_index, uint32 labels_to_evaluate
  );

  sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  );
  sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation, std::vector<sdouble32>& tmp_data
  );

private:
  static rafko_utilities::DataPool<sdouble32> common_datapool;

  rafko_mainframe::RafkoSettings& settings;
  std::shared_ptr<rafko_gym::CostFunction> cost_function;
  rafko_utilities::ThreadGroup error_calculation_threads;

  /**
   * @brief          Converting the @rafko_gym::DataSet message to vectors. The summary starts at the first element,
   *                 and goes as long as @length defines it
   *
   * @param          source         The vector elements to accumulate into @target
   * @param          target         The value to accept element sum from @source
   * @param[in]      length         The number of elements to take from @source into @target
   * @param          error_mutex    The mutex guarding the update of @target
   * @param[in]      thread_index   The index of the thread the function is called from
   */
  void accumulate_error_sum(std::vector<sdouble32>& source, sdouble32& target, uint32 length, std::mutex& error_mutex, uint32 thread_index);

};

} /* namespace rafko_gym */

#endif /* RAFKO_DATASET_COST_H */
