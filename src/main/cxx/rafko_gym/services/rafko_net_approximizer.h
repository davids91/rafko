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

#include <cmath>
#include <vector>
#include <limits>

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_net/services/updater_factory.h"
#include "rafko_net/services/rafko_weight_updater.h"

#include "rafko_gym/services/rafko_agent.h"
#include "rafko_gym/services/rafko_environment.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

using std::min;
using std::vector;
using std::unique_ptr;

using rafko_mainframe::RafkoServiceContext;
using rafko_net::RafkoNet;
using rafko_net::SolutionBuilder;
using rafko_net::SolutionSolver;
using rafko_net::RafkoWeightUpdater;
using rafko_net::Weight_updaters;
using rafko_net::GradientFragment;
using rafko_net::UpdaterFactory;

/**
 * @brief      This class approximates gradients for a @Dataset and @RafkoNet.
 *             The approximated gradients are collected into one gradient fragment.
 */
class RafkoNetApproximizer{
public:

  /**
   * @brief      Class Constructor
   *
   * @param      service_context_              The service context in which the object should be executed
   * @param[in]  neural_network                The Network to optimize based on the gradient approximation
   * @param      environment_                  The Data Environment the network should be evaluated in
   * @param[in]  weight_updater_               The Weight updater to help convergence
   * @param[in]  stochastic_evaluation_loops_  Decideshow many stochastic evaluations of the @neural_network shall count as one evaluation during gradient approximation
   */
  RafkoNetApproximizer(
    RafkoServiceContext& service_context_, RafkoNet& neural_network, RafkoEnvironment& environment_,
    Weight_updaters weight_updater_, uint32 stochastic_evaluation_loops_ = 1u
  ):service_context(service_context_)
  , net(neural_network)
  , net_solution(SolutionBuilder(service_context).build(net))
  , environment(environment_)
  , solver(SolutionSolver::Builder(*net_solution, service_context).build())
  , weight_updater(UpdaterFactory::build_weight_updater(net, *net_solution, weight_updater_, service_context))
  , stochastic_evaluation_loops(stochastic_evaluation_loops_)
  , applied_direction(net.weight_table_size())
  {
    environment.full_evaluation(*solver);
  }

  ~RafkoNetApproximizer(void){
    if(nullptr == service_context.get_arena_ptr())
      delete net_solution;
  }
  RafkoNetApproximizer(const RafkoNetApproximizer& other) = delete;/* Copy constructor */
  RafkoNetApproximizer(RafkoNetApproximizer&& other) = delete; /* Move constructor */
  RafkoNetApproximizer& operator=(const RafkoNetApproximizer& other) = delete; /* Copy assignment */
  RafkoNetApproximizer& operator=(RafkoNetApproximizer&& other) = delete; /* Move assignment */

  /**
   * @brief      Moves the network in a direction based on induvidual weight gradients,
   *             approximates the gradients based on that and then reverts the the weight change
   */
  void collect_approximates_from_weight_gradients(void);

  /**
   * @brief      Move the network in the given direction, collect approximate gradient for it
   *             and then reverts the weight change
   *
   * @param      direction         The direction
   * @param[in]  save_to_fragment  Decides wether or not to add the results into the collected gradient fragments
   */
  void convert_direction_to_gradient(vector<sdouble32>& direction, bool save_to_fragment);

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
  sdouble32 get_gradient_for_all_weights(void);

  /**
   * @brief      Applies the colleted gradient fragment to the configured network
   */
  void apply_fragment(void);

  /**
   * @brief      Discards the gradient fragment collected in the past
   */
  void discard_fragment(void){
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
  const GradientFragment get_fragment(void){
    return gradient_fragment;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  const GradientFragment& get_weight_gradient(void) const{
    return gradient_fragment;
  }

  /**
   * @brief      Evaluates the network in the given environment fully
   */
  void full_evaluation(void){
    environment.full_evaluation(*solver);
    if(min_test_error > environment.get_testing_fitness()){
      min_test_error = environment.get_testing_fitness();
      min_test_error_was_at_iteration = iteration;
    }
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  bool stop_training(void){
    return(
      (1u < iteration)
      &&((
        (
          service_context.get_training_strategy(rafko_net::Training_strategy::training_strategy_stop_if_training_error_below_learning_rate)
          &&(service_context.get_learning_rate() >= -environment.get_training_fitness())
        )||(
          service_context.get_training_strategy(rafko_net::Training_strategy::training_strategy_stop_if_training_error_zero)
          &&(double_literal(0.0) ==  -environment.get_training_fitness())
        )
      )||(
        service_context.get_training_strategy(rafko_net::Training_strategy::training_strategy_early_stopping)
        &&(environment.get_testing_fitness() < (min_test_error - (min_test_error * service_context.get_delta())))
        &&((iteration - min_test_error_was_at_iteration) > service_context.get_tolerance_loop_value())
      ))
    );
  }

private:
  RafkoServiceContext& service_context;
  RafkoNet& net;
  Solution* net_solution;
  RafkoEnvironment& environment;
  unique_ptr<RafkoAgent> solver;
  unique_ptr<RafkoWeightUpdater> weight_updater;
  GradientFragment gradient_fragment;
  uint32 stochastic_evaluation_loops;

  uint32 iteration = 1;
  vector<sdouble32> applied_direction;
  sdouble32 epsilon_addition = double_literal(0.0);
  sdouble32 min_test_error = std::numeric_limits<sdouble32>::max();
  uint32 min_test_error_was_at_iteration = 0;

  /**
   * @brief      Evaluates the network in a stochastic manner the number of configured times and return with the fittness/error value
   *
   * @return         The average of the resulting fitness values of the evaluations
   */
  sdouble32 stochastic_evaluation(void){
    sdouble32 fitness = double_literal(0.0);
    for(uint32 i = 0; i < stochastic_evaluation_loops; ++i)
      fitness += environment.stochastic_evaluation(*solver, iteration);
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
