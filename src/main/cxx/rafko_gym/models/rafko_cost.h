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

#ifndef RAFKO_COST_H
#define RAFKO_COST_H

#include "rafko_global.h"

#include <vector>
#include <memory>

#include "rafko_utilities/services/thread_group.h"
#include "rafko_utilities/models/data_pool.h"
#include "rafko_gym/services/cost_function.h"
#include "rafko_gym/services/function_factory.h"
#include "rafko_mainframe/models/rafko_settings.h"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_agent.h"

namespace rafko_gym{

/**
 * @brief      Error Statistics for a @RafkoDatasetWrapper
 */
class RAFKO_FULL_EXPORT RafkoCost : public RafkoObjective
{
public:
  RafkoCost(rafko_mainframe::RafkoSettings& settings_, std::shared_ptr<rafko_gym::CostFunction> cost_function_)
  : settings(settings_)
  , cost_function(cost_function_)
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  RafkoCost(rafko_mainframe::RafkoSettings& settings_, rafko_gym::Cost_functions the_function)
  : settings(settings_)
  , cost_function(rafko_gym::FunctionFactory::build_cost_function(the_function, settings_))
  , error_calculation_threads(settings_.get_sqrt_of_solve_threads())
  { }

  double set_feature_for_label(
    const rafko_gym::RafkoEnvironment& environment, std::uint32_t sample_index,
    const std::vector<double>& neuron_data
  ) const;

  double set_features_for_labels(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t raw_start_index, std::uint32_t labels_to_evaluate
  ) const;

  double set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
    std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation
  ) const;
  double set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& environment, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
    std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation, std::vector<double>& tmp_data
  ) const;

  #if(RAFKO_USES_OPENCL)
  void set_gpu_parameters(std::uint32_t pairs_to_evaluate_, std::uint32_t feature_size_){
    cost_function->set_parameters(pairs_to_evaluate_, feature_size_);
    pairs_to_evaluate = pairs_to_evaluate_;
  }
  cl::Program::Sources get_step_sources() const{
    return cost_function->get_step_sources();
  }
  std::vector<std::string> get_step_names() const{
    return cost_function->get_step_names();
  }

  /**
   * @brief      Provides the input dimensions of the objective, consiting of
   *             3 buffers in 2 phases:
   *              1. input shapes coming from the cost function
   *              2. an error array for each pair
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const{
    return cost_function->get_input_shapes();
  }

  /**
   * @brief      Provides the output dimensions of the objective, consiting of
   *             2 buffers in 2 phases:
   *              1. output shapes coming from the cost function
   *              2. a resulting error value
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const{
    return cost_function->get_output_shapes();
  }

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const{
    return cost_function->get_solution_space();
  }
  #endif/*(RAFKO_USES_OPENCL)*/


private:
  static rafko_utilities::DataPool<double> common_datapool;

  rafko_mainframe::RafkoSettings& settings;
  std::shared_ptr<rafko_gym::CostFunction> cost_function;
  rafko_utilities::ThreadGroup error_calculation_threads;
  #if(RAFKO_USES_OPENCL)
  std::uint32_t pairs_to_evaluate = 1u;
  #endif/*(RAFKO_USES_OPENCL)*/

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
  void accumulate_error_sum(std::vector<double>& source, double& target, std::uint32_t length, std::mutex& error_mutex, std::uint32_t thread_index) const;

};

} /* namespace rafko_gym */

#endif /* RAFKO_COST_H */
