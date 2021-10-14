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

#ifndef SPARSE_NET_APPROXIMIZER_H
#define SPARSE_NET_APPROXIMIZER_H

#include "rafko_global.h"

#include <cmath>
#include <vector>

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/sparse_net.pb.h"

#include "rafko_mainframe/models/service_context.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_net/services/updater_factory.h"
#include "rafko_net/services/weight_updater.h"

#include "rafko_gym/services/agent.h"
#include "rafko_gym/services/environment.h"

namespace rafko_gym{

using std::min;
using std::vector;
using std::unique_ptr;

using rafko_mainframe::Service_context;
using rafko_net::SparseNet;
using rafko_net::Solution_builder;
using rafko_net::Solution_solver;
using rafko_net::Weight_updater;
using rafko_net::weight_updaters;
using rafko_net::Gradient_fragment;
using rafko_net::Updater_factory;

/**
 * @brief      This class approximates gradients for a @Dataset and @Sparse_net.
 *             The approximated gradients are collected into one gradient fragment.
 */
class Sparse_net_approximizer{
public:

  Sparse_net_approximizer(Service_context& service_context_, SparseNet& neural_network, Environment& environment_, weight_updaters weight_updater_)
  : service_context(service_context_)
  , net(neural_network)
  , net_solution(Solution_builder(service_context).build(net))
  , environment(environment_)
  , solver(Solution_solver::Builder(*net_solution, service_context).build())
  , applied_direction(net.weight_table_size())
  {
    weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,service_context);
    environment.full_evaluation(*solver);
  }

  ~Sparse_net_approximizer(void){
    if(nullptr == service_context.get_arena_ptr())
      delete net_solution;
  }
  Sparse_net_approximizer(const Sparse_net_approximizer& other) = delete;/* Copy constructor */
  Sparse_net_approximizer(Sparse_net_approximizer&& other) = delete; /* Move constructor */
  Sparse_net_approximizer& operator=(const Sparse_net_approximizer& other) = delete; /* Copy assignment */
  Sparse_net_approximizer& operator=(Sparse_net_approximizer&& other) = delete; /* Move assignment */

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
    gradient_fragment = Gradient_fragment();
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
  const Gradient_fragment get_fragment(void){
    return gradient_fragment;
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  const Gradient_fragment& get_weight_gradient(void) const{
    return gradient_fragment;
  }

private:
  Service_context& service_context;
  SparseNet& net;
  Solution* net_solution;
  Environment& environment;
  unique_ptr<Agent> solver;
  unique_ptr<Weight_updater> weight_updater;
  Gradient_fragment gradient_fragment;

  uint32 iteration = 1;
  vector<sdouble32> applied_direction;

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

#endif /* SPARSE_NET_APPROXIMIZER_H */
