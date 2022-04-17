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
#include <iostream>
#include <iomanip>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_gym/models/rafko_cost.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_gym/services/rafko_backpropagation.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

TEST_CASE("Testing if autodiff optimizer converges small 1 Neuron networks", "[optimize][small]"){
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_learning_rate(8e-2).set_minibatch_size(64).set_memory_truncation(2)
    .set_droput_probability(0.2)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero,true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping,false)
    .set_learning_rate_decay({{1000u,0.8}})
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4);

  rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
    .input_size(2).expected_input_range((1.0))
    .add_feature_to_layer(1, rafko_net::neuron_group_feature_boltzmann_knot)
    // .add_feature_to_layer(0, rafko_net::neuron_group_feature_l1_regularization)
    .add_feature_to_layer(0, rafko_net::neuron_group_feature_l2_regularization)
    // // .add_feature_to_layer(1, rafko_net::neuron_group_feature_l1_regularization)
    .add_feature_to_layer(1, rafko_net::neuron_group_feature_l2_regularization)
    // // .add_feature_to_layer(2, rafko_net::neuron_group_feature_l1_regularization)
    .add_feature_to_layer(2, rafko_net::neuron_group_feature_l2_regularization)
    // .add_feature_to_layer(1, rafko_net::neuron_group_feature_dropout_regularization)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_identity},
    }).dense_layers({1});
    //TODO: single input Neuron, no transfer function, no spike function

  rafko_gym::RafkoBackPropagation optimizer(*network, settings);


  //TODO: Build Environment nd Objective
}

} /* namespace rafko_gym_test */
