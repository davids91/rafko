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

#include <math.h>
#include <vector>
#include <limits>

#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_context.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"

#include "rafko_gym/services/updater_factory.h"
#include "rafko_gym/services/rafko_weight_updater.h"
#include "rafko_gym/models/rafko_agent.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      This class approximates gradients for a @Dataset and @RafkoNet.
 *             The approximated gradients are collected into one gradient fragment.
 */
class RafkoNetApproximizer{
public:

  /**
   * @brief      Class Constructor
   *
   * @param      context_                      The service context containing the network to enchance
   * @param[in]  stochastic_evaluation_loops_  Decideshow many stochastic evaluations of the @neural_network shall count as one evaluation during gradient approximation
   */
  RafkoNetApproximizer(rafko_mainframe::RafkoContext& context_, uint32 stochastic_evaluation_loops_ = 1u)
  : context(context_)
  , stochastic_evaluation_loops(stochastic_evaluation_loops_)
  , applied_direction(context.expose_network().weight_table_size())
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
  void convert_direction_to_gradient(std::vector<sdouble32>& direction, bool save_to_fragment);

  /**
   * @brief      Collects the approximate gradient of a single weight
   *
   * @param[in]  weight_index  The weight index to approximate for
   *
   * @return     The gradient approximation for the configured dataset
   */
  sdouble32 get_single_weight_gradient(uint32 weight_index);

  /**
   * @brief      APproximates gradient information for all weights.
   *
   * @return     The gradient for all weights.
   */
  sdouble32 get_gradient_for_all_weights();

  /**
   * @brief      Applies the colleted gradient fragment to the configured network
   */
  void apply_fragment();

  /**
   * @brief      Discards the gradient fragment collected in the past
   */
  void discard_fragment(){
    gradient_fragment = GradientFragment();
  }

  /**
   * @brief      Adds the given values to the stored fragment.
   *
   * @param[in]  weight_index             The weight index to give the value to
   * @param[in]  gradient_fragment_value  The value to give to the fragment
   */
  void add_to_fragment(uint32 weight_index, sdouble32 gradient_fragment_value);

  /**
   * @brief      Gets the previously collected gradient fragment.
   *
   * @return     The fragment.
   */
  const GradientFragment get_fragment(){
    return gradient_fragment;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  const GradientFragment& get_weight_gradient() const{
    return gradient_fragment;
  }

  /**
   * @brief      Evaluates the network in the given environment fully
   */
  void full_evaluation(){
    context.full_evaluation();
    if(min_test_error > context.get_objective().get_feature_fitness()){
      min_test_error = context.get_objective().get_feature_fitness();
      min_test_error_was_at_iteration = iteration;
    }
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
          &&(context.expose_settings().get_learning_rate() >= -context.get_objective().get_feature_fitness())
        )||(
          context.expose_settings().get_training_strategy(Training_strategy::training_strategy_stop_if_training_error_zero)
          &&(double_literal(0.0) ==  -context.get_objective().get_feature_fitness())
        )
      ))
    );
  }

private:
  rafko_mainframe::RafkoContext& context;
  GradientFragment gradient_fragment;
  uint32 stochastic_evaluation_loops;

  uint32 iteration = 1;
  std::vector<sdouble32> applied_direction;
  sdouble32 epsilon_addition = double_literal(0.0);
  sdouble32 min_test_error = std::numeric_limits<sdouble32>::max();
  uint32 min_test_error_was_at_iteration = 0;

  /**
   * @brief      Evaluates the network in a stochastic manner the number of configured times and return with the fittness/error value
   *
   * @return         The average of the resulting fitness values of the evaluations
   */
  sdouble32 stochastic_evaluation(){
    sdouble32 fitness = double_literal(0.0);
    for(uint32 i = 0; i < stochastic_evaluation_loops; ++i)
      fitness += context.stochastic_evaluation(iteration);
    return fitness / static_cast<sdouble32>(stochastic_evaluation_loops);
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
  static void insert_element_at_position(google::protobuf::RepeatedField<sdouble32>& message_field, sdouble32 value, uint32 position){
    *message_field.Add() = value;
    for(sint32 i(message_field.size() - 1); i > static_cast<sint32>(position); --i)
      message_field.SwapElements(i, i - 1);
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_NET_APPROXIMIZER_H */
