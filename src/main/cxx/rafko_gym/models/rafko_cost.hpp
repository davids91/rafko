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

#include "rafko_global.hpp"

#include <memory>
#include <vector>

#include "rafko_gym/services/cost_function.hpp"
#include "rafko_gym/services/function_factory.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_utilities/models/data_pool.hpp"
#include "rafko_utilities/services/thread_group.hpp"
#if (RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#endif /*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/models/rafko_objective.hpp"

namespace rafko_gym {

/**
 * @brief      Implementation of an objective based on a @CostFunction
 */
class RAFKO_EXPORT RafkoCost : public virtual RafkoObjective {
public:
  RafkoCost(rafko_mainframe::RafkoSettings &settings,
            std::shared_ptr<rafko_gym::CostFunction> cost_function)
      : m_settings(settings), m_costFunction(cost_function),
        m_errorCalculationThreads(settings.get_sqrt_of_solve_threads()) {}

  RafkoCost(rafko_mainframe::RafkoSettings &settings,
            rafko_gym::Cost_functions the_function)
      : m_settings(settings),
        m_costFunction(rafko_gym::FunctionFactory::build_cost_function(
            the_function, settings)),
        m_errorCalculationThreads(settings.get_sqrt_of_solve_threads()) {}

  ~RafkoCost() = default;

  /* +++ Methods taken from @RafkoObjective +++ */
  Cost_functions get_cost_type() const override {
    return m_costFunction->get_type();
  }

  double
  set_feature_for_label(const rafko_gym::RafkoDataSet &environment,
                        std::uint32_t sample_index,
                        const std::vector<double> &neuron_data) const override;

  double
  set_features_for_labels(const rafko_gym::RafkoDataSet &environment,
                          const std::vector<std::vector<double>> &neuron_data,
                          std::uint32_t neuron_buffer_index,
                          std::uint32_t raw_start_index,
                          std::uint32_t labels_to_evaluate) const override;

  double set_features_for_sequences(
      const rafko_gym::RafkoDataSet &environment,
      const std::vector<std::vector<double>> &neuron_data,
      std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index,
      std::uint32_t sequences_to_evaluate,
      std::uint32_t start_index_in_sequence,
      std::uint32_t sequence_truncation) const override;

  double set_features_for_sequences(
      const rafko_gym::RafkoDataSet &environment,
      const std::vector<std::vector<double>> &neuron_data,
      std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index,
      std::uint32_t sequences_to_evaluate,
      std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation,
      std::vector<double> &tmp_data) const override;

  double get_derivative(double label_value, double feature_value,
                        double feature_d, double sample_number) const override {
    return m_costFunction->get_derivative(label_value, feature_value, feature_d,
                                          sample_number);
  }

#if (RAFKO_USES_OPENCL)
  void set_gpu_parameters(std::uint32_t pairs_to_evaluate,
                          std::uint32_t feature_size) override {
    m_costFunction->set_parameters(pairs_to_evaluate, feature_size);
    m_pairsToEvaluate = pairs_to_evaluate;
  }

  /**
   * @brief      Provides the kernel function for the derivative of all of the
   * cost functions
   *
   * @param[in]  label_value        The label value
   * @param[in]  feature_value      The data to comapre to the label value
   * @param[in]  feature_d          The derivative of the of the feature value
   * @param[in]  sample_number      The number of sample values the objective is
   * evaluated on at once
   * @param[in]  target             The variable to store the result of the
   * instructions in
   * @param[in]  behavior_index     The value corresponding to the cost function
   * (@get_kernel_enum in rafko_gym::CostFunction)
   *
   * @return     The source for implementing the kernel of the derivative of the
   * cost function
   */
  static std::string generic_derivative_kernel_source(
      std::string label_value, std::string feature_value, std::string feature_d,
      std::string sample_number, std::string target,
      std::string behavior_index);

  cl::Program::Sources get_step_sources() const override {
    return m_costFunction->get_step_sources();
  }

  std::vector<std::string> get_step_names() const override {
    return m_costFunction->get_step_names();
  }

  /**
   * @brief      Provides the input dimensions of the objective, consiting of
   *             3 buffers in 2 phases:
   *              1. input shapes coming from the cost function
   *              2. an error array for each pair
   *
   * @return     Vector of dimensions in order of @get_step_sources and
   * @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape>
  get_input_shapes() const override {
    return m_costFunction->get_input_shapes();
  }

  /**
   * @brief      Provides the output dimensions of the objective, consiting of
   *             2 buffers in 2 phases:
   *              1. output shapes coming from the cost function
   *              2. a resulting error value
   *
   * @return     Vector of dimensions in order of @get_step_sources and
   * @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape>
  get_output_shapes() const override {
    return m_costFunction->get_output_shapes();
  }

  std::tuple<cl::NDRange, cl::NDRange, cl::NDRange>
  get_solution_space() const override {
    return m_costFunction->get_solution_space();
  }
#endif /*(RAFKO_USES_OPENCL)*/
  /* --- Methods taken from @RafkoObjective --- */

private:
  static rafko_utilities::DataPool<double> m_commonDatapool;

  rafko_mainframe::RafkoSettings &m_settings;
  std::shared_ptr<rafko_gym::CostFunction> m_costFunction;
  rafko_utilities::ThreadGroup m_errorCalculationThreads;
#if (RAFKO_USES_OPENCL)
  std::uint32_t m_pairsToEvaluate = 1u;
#endif /*(RAFKO_USES_OPENCL)*/

  /**
   * @brief          Converting the @rafko_gym::DataSetPackage message to
   * vectors. The summary starts at the first element, and goes as long as
   * @length defines it
   *
   * @param          source         The vector elements to accumulate into
   * @target
   * @param          target         The value to accept element sum from @source
   * @param[in]      length         The number of elements to take from @source
   * into @target
   * @param          error_mutex    The mutex guarding the update of @target
   * @param[in]      thread_index   The index of the thread the function is
   * called from
   */
  void accumulate_error_sum(std::vector<double> &source, double &target,
                            std::uint32_t length, std::mutex &error_mutex,
                            std::uint32_t thread_index) const;
};

} /* namespace rafko_gym */

#endif /* RAFKO_COST_H */
