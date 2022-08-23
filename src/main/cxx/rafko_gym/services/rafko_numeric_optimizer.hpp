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

#ifndef RAFKO_NUMERIC_OPTIMIZER_H
#define RAFKO_NUMERIC_OPTIMIZER_H

#include "rafko_global.hpp"

#include <memory>
#include <vector>
#include <limits>
#include <mutex>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_context.hpp"

namespace rafko_gym{

/**
 * @brief      This class approximates gradients for a @Dataset and @RafkoNet.
 *             The approximated gradients are collected into one gradient fragment.
 */
class RAFKO_FULL_EXPORT RafkoNumericOptimizer{
public:

  /**
   * @brief      Class Constructor
   *
   * @param      contexts                     An array of service contexts to use for training the network
   * @param[in]  stochastic_evaluation_loops  Decideshow many stochastic evaluations of the @neural_network shall count as one evaluation during gradient approximation
   */
  RafkoNumericOptimizer(
    std::vector<std::shared_ptr<rafko_mainframe::RafkoContext>> contexts,
    std::shared_ptr<rafko_mainframe::RafkoContext> test_context,
    rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings(),
    std::uint32_t stochastic_evaluation_loops = 1u
  ):settings(settings)
  , training_contexts(contexts)
  , test_context(test_context)
  , weight_filter(training_contexts[0]->expose_network().weight_table_size(), 1.0)
  , used_weight_filter(weight_filter)
  , weight_exclude_chance_filter(training_contexts[0]->expose_network().weight_table_size(), 0.0)
  , stochastic_evaluation_loops(stochastic_evaluation_loops)
  , execution_threads(std::min(training_contexts.size(), static_cast<std::size_t>(settings.get_max_processing_threads())))
  , m_tmpDataPool(2u, training_contexts[0]->expose_network().weight_table_size())
  { }

  RafkoNumericOptimizer(const RafkoNumericOptimizer& other) = delete;/* Copy constructor */
  RafkoNumericOptimizer(RafkoNumericOptimizer&& other) = delete; /* Move constructor */
  RafkoNumericOptimizer& operator=(const RafkoNumericOptimizer& other) = delete; /* Copy assignment */
  RafkoNumericOptimizer& operator=(RafkoNumericOptimizer&& other) = delete; /* Move assignment */

  /**
   * @brief      Moves the network in a direction based on induvidual weight gradients,
   *             approximates the gradients based on that and then reverts the the weight change
   */
  void collect_approximates_from_weight_gradients();

  /**
   * @brief      Move the network in the given direction, collect approximate gradient for it
   *             and then reverts the weight change
   *
   * @param      direction         The direction
   * @param[in]  save_to_fragment  Decides wether or not to add the results into the collected gradient fragments
   */
  void convert_direction_to_gradient(std::vector<double>& direction, bool save_to_fragment);

  /**
   * @brief      Collects the approximate gradient of a single weight
   *
   * @param[in]  weight_index  The weight index to approximate for
   * @param      context       Reference to the context handling the evaluation for the gradients
   *
   * @return     The gradient approximation for the configured dataset
   */
  double get_single_weight_gradient(std::uint32_t weight_index, rafko_mainframe::RafkoContext& context);

  /**
   * @brief      APproximates gradient information for all weights.
   *
   * @return     The gradient for all weights.
   */
  double get_gradient_for_all_weights();

  /**
   * @brief      Applies the colleted gradient fragment to the configured network
   */
  void apply_weight_vector_delta();

  /**
   * @brief      Discards the gradient fragment collected in the past
   */
  void discard_fragment(){
    gradient_fragment = NetworkWeightVectorDelta();
  }

  /**
   * @brief      Adds the given values to the stored fragment.
   *
   * @param[in]  weight_index             The weight index to give the value to
   * @param[in]  gradient_fragment_value  The value to give to the fragment
   */
  void add_to_fragment(std::uint32_t weight_index, double gradient_fragment_value);

  /**
   * @brief      Gets the previously collected gradient fragment.
   *
   * @return     The fragment.
   */
  const NetworkWeightVectorDelta get_fragment(){
    return gradient_fragment;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  constexpr const NetworkWeightVectorDelta& get_weight_gradient() const{
    return gradient_fragment;
  }

  /**
   * @brief   Function to set weight of the networks weight in bulk for iterations;
   *          Filter vector must be equal in sze of the stored networks weight table
   *          Meaning: no modification | 0 --> 1 | gradient * 1.0
   *
   * @param[in]   filter    the filter to set for the whole weight array
   */
  void set_weight_filter(std::vector<double>&& filter){
    RFASSERT_LOG("Weight filter size: {} vs. {}", weight_filter.size(), filter.size());
    RFASSERT( filter.size() == weight_filter.size());
    weight_filter = filter;
  }

  /**
   * @brief   Function to set weight of the networks weight for iterations;
   *          Filter vector must be equal in sze of the stored networks weight table
   *          Meaning: no modification | 0 --> 1 | gradient * 1.0
   *
   * @param[in]   weight_index    the affected weight index
   * @param[in]   filter          the filter to set for the whole weight array
   */
  void modify_weight_filter(std::uint32_t weight_index, double filter){
    RFASSERT( weight_index < weight_filter.size());
    weight_filter[weight_index] = filter;
  }

  /**
   * @brief     Function to set weight of all of the the networks weight for iterations;
   *            Meaning: no modification | 0 --> 1 | gradient * 1.0
   *
   * @param[in]   filter    the filter to set for the whole weight filter array
   */
  void set_weight_filter(double filter){
    std::fill(weight_filter.begin(),weight_filter.end(), filter);
  }

  /**
   * @brief   Function to set weight chance to be excluded in bulk for iterations;
   *          Filter vector must be equal in sze of the stored networks weight table
   *          Meaning: don't exclude weight: 0 --> 1 definitely exclude weight
   *
   * @param[in]   filter    the filter to set for the whole weight array
   */
  void set_weight_exclude_chance_filter(std::vector<double>&& filter){
    RFASSERT( filter.size() == weight_exclude_chance_filter.size());
    weight_exclude_chance_filter = filter;
    m_excludeChanceSum = std::accumulate(
      weight_exclude_chance_filter.begin(),weight_exclude_chance_filter.end(), 0.0
    );
  }

  /**
   * @brief   Function to set weight chance to be excluded in bulk for iterations;
   *          The given value shall be set for all weights
   *          Meaning: don't exclude weight: 0 --> 1 definitely exclude weight
   *
   * @param[in]   filter    the filter to set for the whole weight array
   */
  void set_weight_exclude_chance_filter(double filter){
    std::fill(weight_exclude_chance_filter.begin(),weight_exclude_chance_filter.end(), filter);
    m_excludeChanceSum = filter * static_cast<double>(weight_exclude_chance_filter.size());
  }

  /**
   * @brief   Function to set weight chance to be excluded in bulk for iterations;
   *          Filter vector must be equal in sze of the stored networks weight table
   *          Meaning: don't exclude weight: 0 --> 1 definitely exclude weight
   *
   * @param[in]   weight_index    the affected weight index
   * @param[in]   filter          the filter to set for the whole weight array
   */
  void modify_weight_exclude_chance_filter(std::uint32_t weight_index, double filter){
    RFASSERT( weight_index < weight_exclude_chance_filter.size());
    weight_exclude_chance_filter[weight_index] = filter;
    m_excludeChanceSum = std::accumulate(
      weight_exclude_chance_filter.begin(),weight_exclude_chance_filter.end(), 0.0
    );
  }


  /**
   * @brief      Evaluates the network in the given environment fully
   */
  void full_evaluation(){
    double fitness = training_contexts[0]->full_evaluation();
    if(m_minTestError > fitness){
      m_minTestError = fitness;
      m_minTestErrorWasAtIteration = m_iteration;
    }
    m_errorEstimation = -fitness;
  }

  constexpr double get_error_estimation() const{
    return m_errorEstimation;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  bool stop_training(){
    return(
      (1u < m_iteration)
      &&((
        (
          settings.get_training_strategy(Training_strategy::training_strategy_stop_if_training_error_below_learning_rate)
          &&(settings.get_learning_rate() >= -m_minTestError)
        )||(
          settings.get_training_strategy(Training_strategy::training_strategy_stop_if_training_error_zero)
          &&((0.0) ==  -m_minTestError)
        )||(
          (training_contexts[0] && test_context)
          &&(
            (settings.get_training_strategy(Training_strategy::training_strategy_early_stopping))
            &&(m_lastTrainingError > ( m_lastTestingError * (1.0 + settings.get_delta()) ))
          )
        )
      ))
    );
  }

private:
  rafko_mainframe::RafkoSettings settings;
  std::vector<std::shared_ptr<rafko_mainframe::RafkoContext>> training_contexts;
  std::shared_ptr<rafko_mainframe::RafkoContext> test_context;
  std::vector<double> weight_filter;
  std::vector<double> used_weight_filter;
  std::vector<double> weight_exclude_chance_filter;
  NetworkWeightVectorDelta gradient_fragment;
  std::uint32_t stochastic_evaluation_loops;
  rafko_utilities::ThreadGroup execution_threads;

  std::mutex m_networkMutex;
  std::uint32_t m_iteration = 1u;
  rafko_utilities::DataPool<> m_tmpDataPool;
  double m_epsilonAddition = 0.0;
  double m_minTestError = std::numeric_limits<double>::max();
  double m_lastTrainingError = std::numeric_limits<double>::quiet_NaN();
  double m_lastTestingError = std::numeric_limits<double>::quiet_NaN();
  double m_errorEstimation = 1.0;
  double m_excludeChanceSum = 0.0;
  std::uint32_t m_minTestErrorWasAtIteration = 0u;
  std::uint32_t m_lastTestedIteration = 0u;

  /**
   * @brief      Evaluates the network in a stochastic manner the number of configured times and return with the fittness/error value
   *
   * @return         The average of the resulting fitness values of the evaluations
   */
  double stochastic_evaluation(rafko_mainframe::RafkoContext& context){
    double fitness = (0.0);
    for(std::uint32_t i = 0; i < stochastic_evaluation_loops; ++i)
      fitness += context.stochastic_evaluation(m_iteration);
    double result_fitness = fitness / static_cast<double>(stochastic_evaluation_loops);
    m_errorEstimation = (m_errorEstimation + -result_fitness)/(2.0);
    return result_fitness;
  }

  /**
   * @brief   Calculates the error value from a stochastic evaluation with the network shifted to the given direction
   *          And does it in a thread-safe way for the Original network
   *
   * @param   context                     The used context for stochastic evaluation
   * @param   network_original_weights    The original weights of the network, provided here as an optimiztion step
   * @param   direction                   The direction value to add to every weight of the network
   */
  double get_error_from_direction(
    rafko_mainframe::RafkoContext& context,
    const std::vector<double>& network_original_weights,
    double direction
  );

  /**
   * @brief   Calculates the error value from a stochastic evaluation with the network shifted to the given direction
   *          And does it in a thread-safe way for the Original network
   *
   * @param   context                     The used context for stochastic evaluation
   * @param   network_original_weights    The original weights of the network, provided here as an optimiztion step
   * @param   direction                   The direction value to add to every weight of the network
   */
  double get_error_from_direction(
    rafko_mainframe::RafkoContext& context,
    const std::vector<double>& network_original_weights,
    const std::vector<double>& direction
  );

  /**
   * @brief      Insert an element to the given position into the given field by
   *             first adding it to the end, and then reverse iterating and swapping elements
   *             until the desired position is reached
   *
   * @param      message_field  The message field
   * @param[in]  value          The value
   * @param[in]  position       The position
   */
  static void insert_element_at_position(google::protobuf::RepeatedField<double>& message_field, double value, std::uint32_t position){
    *message_field.Add() = value;
    for(std::int32_t i(message_field.size() - 1); i > static_cast<std::int32_t>(position); --i)
      message_field.SwapElements(i, i - 1);
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_NUMERIC_OPTIMIZER_H */
