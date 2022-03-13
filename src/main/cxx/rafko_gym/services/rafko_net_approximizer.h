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

#ifndef RAFKO_NET_APPROXIMIZER_H
#define RAFKO_NET_APPROXIMIZER_H

#include "rafko_global.h"

#include <vector>
#include <limits>

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_gym{

/**
 * @brief      This class approximates gradients for a @Dataset and @RafkoNet.
 *             The approximated gradients are collected into one gradient fragment.
 */
class RAFKO_FULL_EXPORT RafkoNetApproximizer{
public:

  /**
   * @brief      Class Constructor
   *
   * @param      context_                      The service context containing the network to enchance
   * @param[in]  stochastic_evaluation_loops_  Decideshow many stochastic evaluations of the @neural_network shall count as one evaluation during gradient approximation
   */
  RafkoNetApproximizer(rafko_mainframe::RafkoContext& context_, std::uint32_t stochastic_evaluation_loops_ = 1u)
  : context(context_)
  , weight_filter(context.expose_network().weight_table_size(), 1.0)
  , used_weight_filter(weight_filter)
  , weight_exclude_chance_filter(context.expose_network().weight_table_size(), 0.0)
  , stochastic_evaluation_loops(stochastic_evaluation_loops_)
  , tmp_data_pool(2u, context.expose_network().weight_table_size())
  { }

  RafkoNetApproximizer(const RafkoNetApproximizer& other) = delete;/* Copy constructor */
  RafkoNetApproximizer(RafkoNetApproximizer&& other) = delete; /* Move constructor */
  RafkoNetApproximizer& operator=(const RafkoNetApproximizer& other) = delete; /* Copy assignment */
  RafkoNetApproximizer& operator=(RafkoNetApproximizer&& other) = delete; /* Move assignment */

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
   *
   * @return     The gradient approximation for the configured dataset
   */
  double get_single_weight_gradient(std::uint32_t weight_index);

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

  void set_weight_filter(std::vector<double>&& filter){
    RFASSERT_LOG("Weight filter size: {} vs. {}", weight_filter.size(), filter.size());
    RFASSERT( filter.size() == weight_filter.size());
    weight_filter = filter;
  }

  void modify_weight_filter(std::uint32_t weight_index, double filter){
    RFASSERT( weight_index < weight_filter.size());
    weight_filter[weight_index] = filter;
  }

  void set_weight_exclude_chance_filter(std::vector<double>&& filter){
    RFASSERT( filter.size() == weight_exclude_chance_filter.size());
    weight_exclude_chance_filter = filter;
  }

  void modify_weight_exclude_chance_filter(std::uint32_t weight_index, double filter){
    RFASSERT( weight_index < weight_exclude_chance_filter.size());
    weight_exclude_chance_filter[weight_index] = filter;
  }


  /**
   * @brief      Evaluates the network in the given environment fully
   */
  void full_evaluation(){
    double fitness = context.full_evaluation();
    if(min_test_error > fitness){
      min_test_error = fitness;
      min_test_error_was_at_iteration = iteration;
    }
    error_estimation = -fitness;
  }

  constexpr double get_error_estimation() const{
    return error_estimation;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  bool stop_training(){
    return(
      (1u < iteration)
      &&((
        (
          context.expose_settings().get_training_strategy(Training_strategy::training_strategy_stop_if_training_error_below_learning_rate)
          &&(context.expose_settings().get_learning_rate() >= -min_test_error)
        )||(
          context.expose_settings().get_training_strategy(Training_strategy::training_strategy_stop_if_training_error_zero)
          &&((0.0) ==  -min_test_error)
        )
      ))
    );
  }

private:
  rafko_mainframe::RafkoContext& context;
  std::vector<double> weight_filter;
  std::vector<double> used_weight_filter;
  std::vector<double> weight_exclude_chance_filter;
  NetworkWeightVectorDelta gradient_fragment;
  std::uint32_t stochastic_evaluation_loops;

  std::uint32_t iteration = 1u;
  rafko_utilities::DataPool<> tmp_data_pool;
  double epsilon_addition = (0.0);
  double min_test_error = std::numeric_limits<double>::max();
  double error_estimation = (1.0);
  std::uint32_t min_test_error_was_at_iteration = 0u;

  /**
   * @brief      Evaluates the network in a stochastic manner the number of configured times and return with the fittness/error value
   *
   * @return         The average of the resulting fitness values of the evaluations
   */
  double stochastic_evaluation(){
    double fitness = (0.0);
    for(std::uint32_t i = 0; i < stochastic_evaluation_loops; ++i)
      fitness += context.stochastic_evaluation(iteration);
    double result_fitness = fitness / static_cast<double>(stochastic_evaluation_loops);
    error_estimation = (error_estimation + -result_fitness)/(2.0);
    return result_fitness;
  }

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

#endif /* RAFKO_NET_APPROXIMIZER_H */
