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

#include <set>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test/test_utility.hpp"

#include "rafko_gym/models/rafko_backpropagation_data.hpp"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.hpp"
#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net_test {

using DependencyParams =
    rafko_gym::RafkoBackpropagationOperation::DependencyParameters;
using DependencyRequest =
    rafko_gym::RafkoBackpropagationOperation::DependencyRequest;

namespace {
DependencyRequest call_neuron_input_dep_request(
    const rafko_net::RafkoNet &network, std::uint32_t operation_index,
    std::optional<std::vector<std::uint32_t>> args = std::nullopt) {
  rafko_gym::RafkoBackpropagationData data(network);
  if (args.has_value() && args->size() == 5) {
    return rafko_gym::RafkoBackpropNeuronInputOperation(
               data, network, operation_index, (*args)[0] /*neuron_index*/,
               (*args)[1] /*input_synapse_index*/,
               (*args)[2] /*weight_synapse_index*/,
               (*args)[3] /*start_inside_input_synapse*/,
               (*args)[4] /*start_inside_weight_synapse*/)
        .request_dependencies();
  } else {
    return rafko_gym::RafkoBackpropNeuronInputOperation(
               data, network, operation_index, 0u /*neuron_index*/)
        .request_dependencies();
  }
}
} // namespace

/*==============================================================================
 * Testing index values of tnext dependencies
 *==============================================================================
 **/
TEST_CASE("Testing if Neuron input dependencies are generating correct "
          "dependency requests for a contigous whole synapse; For network "
          "inputs only",
          "[optimizer][back-propagation][neuron-input]") {
  /* Construct one single neuron with a simple contigous synapse */
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);
  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size + 1); /* spike weight + inputs */

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* No dependency should be required */
  REQUIRE(!dependency_requests.has_value());
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for a contigous whole synapse with one bias value included; For "
    "network inputs only",
    "[optimizer][back-propagation][neuron-input][bias]") {
  /* Construct one single neuron with a simple contigous synapse */
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);
  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size + 2); /* spike weight + inputs + 1 bias weight*/

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* Only the next bias dependency should be required, which points to the bias
   * value */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first == rafko_gym::ad_operation_neuron_bias_d);

  /* The Bias dependency should be constructed for neuron 0 and weight 11 inside
   * the neuron. The dependency is constructed by the Autodiff optimizer,
   * providing the first 3 arguments(data, network and operation index). All
   * other arguments are provided by the dependency request
   */
  REQUIRE(dep_params_vec.back().second[0] == 0u); /* neuron index */
  REQUIRE(dep_params_vec.back().second[1] ==
          synapse_size + 1u); /* neuron weight index */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for a contigous half synapse with one bias value included; For "
    "network inputs only",
    "[optimizer][back-propagation][neuron-input][bias]") {

  /* Construct one single neuron with a simple contigous synapse */
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1 - synapse_size / 2);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 2);
  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2 + 2); /* spike weight + inputs + 1 bias weight*/

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* Only the next bias dependency should be required, which points to the bias
   * value */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first == rafko_gym::ad_operation_neuron_bias_d);

  /* The Bias dependency should be constructed for neuron 0 and weight 11 inside
   * the neuron. The dependency is constructed by the Autodiff optimizer,
   * providing the first 3 arguments(data, network and operation index). All
   * other arguments are provided by the dependency request
   */
  REQUIRE(dep_params_vec.back().second[0] == 0u); /* neuron index */
  REQUIRE(dep_params_vec.back().second[1] ==
          synapse_size / 2u + 1u); /* neuron weight index */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where input synapse is cut in the middle; For "
    "network inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 2u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(-1 - synapse_size / 2);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 2u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2 + 2); /* spike weight + inputs + 1 bias weight*/

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  /* The dependency should be constructed for neuron 0 */
  REQUIRE(dep_params_vec.back().second[0] == 0u); /* neuron index */
  /* The dependency should point to the second input synapse */
  REQUIRE(dep_params_vec.back().second[1] == 1u); /* input_synapse_index */
  /* The dependency should point to the first weight synapse */
  REQUIRE(dep_params_vec.back().second[2] == 0u); /* weight_synapse_index */
  /* The dependency start should point to the start of the input synapse */
  REQUIRE(dep_params_vec.back().second[3] ==
          0u); /* start_inside_input_synapse */
  /* The dependency start should point to the half of the weight synapse + 1 */
  REQUIRE(dep_params_vec.back().second[4] ==
          synapse_size / 2u + 1u); /* start_inside_weight_synapse */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where bias is included and where weight "
    "synapse is cut in the middle; For network inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u + 1u); /* spike weight + half of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 2u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 2u + 1u); /* bias weight + half of the inputs */

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  /* The dependency should be constructed for neuron 0 */
  REQUIRE(dep_params_vec.back().second[0] == 0u); /* neuron index */
  /* The dependency should point to the first input synapse */
  REQUIRE(dep_params_vec.back().second[1] == 0u); /* input_synapse_index */
  /* The dependency should point to the second weight synapse */
  REQUIRE(dep_params_vec.back().second[2] == 1u); /* weight_synapse_index */
  /* The dependency start should point to the half of the input synapse */
  REQUIRE(dep_params_vec.back().second[3] ==
          synapse_size / 2u); /* start_inside_input_synapse */
  /* The dependency start should point to the start of the weight synapse */
  REQUIRE(dep_params_vec.back().second[4] ==
          0); /* start_inside_weight_synapse */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses with one bias included, where weight synapse is "
    "cut in the middle; For network inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 2);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(synapse_size / 2u);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u +
      2u); /* spike weight + half of the inputs + bias weight */

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first == rafko_gym::ad_operation_neuron_bias_d);

  /* The Bias dependency should be constructed for neuron 0 */
  REQUIRE(dep_params_vec.back().second[0] == 0u); /* neuron index */
  REQUIRE(dep_params_vec.back().second[1] ==
          synapse_size / 2u + 1u); /* neuron weight index */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses with one bias included, where weight synapse is "
    "cut two times, Input synapse is cut in the middle; For network inputs "
    "only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 12u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 2u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(-1 - synapse_size / 2);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 2u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 3u + 1u); /* spike weight + third of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 3u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 3u); /* third of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(2)->set_starts(2u * synapse_size / 3u + 1u);
  neuron.mutable_input_weights(2)->set_interval_size(
      synapse_size / 3u + 1u); /* bias weight + third of the inputs */

  /* Construct first Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep = dep_params_vec.back().second;
  REQUIRE(next_dep[0] == 0u);                /* neuron index */
  REQUIRE(next_dep[1] == 0u);                /* input_synapse_index */
  REQUIRE(next_dep[2] == 1u);                /* weight_synapse_index */
  REQUIRE(next_dep[3] == synapse_size / 3u); /* start_inside_input_synapse */
  REQUIRE(next_dep[4] == 0u);                /* start_inside_weight_synapse */

  /* Construct the second part of the synapses, check the third */
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep});

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec2 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec2.size());
  REQUIRE(dep_params_vec2.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep2 = dep_params_vec2.back().second;
  REQUIRE(next_dep2[0] == 0u); /* neuron index */
  REQUIRE(next_dep2[1] == 1u); /* input_synapse_index */
  REQUIRE(next_dep2[2] == 1u); /* weight_synapse_index */
  REQUIRE(next_dep2[3] == 0u); /* start_inside_input_synapse */
  REQUIRE(next_dep2[4] ==
          synapse_size / 3u / 2u); /* start_inside_weight_synapse */

  /* The one after the next dependency should point to the last part of the
   * Neuron input before a bias operation */
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep2});
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec3 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec3.size());
  REQUIRE(dep_params_vec3.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep3 = dep_params_vec3.back().second;
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep3});

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec4 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec4.size());
  REQUIRE(dep_params_vec4.back().first ==
          rafko_gym::ad_operation_neuron_bias_d);

  const auto &next_dep4 = dep_params_vec3.back().second;
  REQUIRE(next_dep4[0] == 0u);                /* neuron index */
  REQUIRE(next_dep4[1] == synapse_size + 1u); /* neuron_weight_index */
  /*!Note: Biasoperation takes the weight synapse index, in this case it's
   * spike weight + inputs + bias */
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses with one bias included, where weight synapse is "
    "cut in the middle, Input synapse is cut two times; No intervals start "
    "from the beginning; For network inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 12u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(-1 - synapse_size / 3);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(2)->set_starts(-1 - 2 * synapse_size / 3);
  neuron.mutable_input_indices(2)->set_interval_size(synapse_size / 3u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u + 1u); /* spike weight + half of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 2u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 2u + 1u); /* half of the inputs + bias weight */

  /* Construct first Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(1u == dep_params_vec.size());
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep = dep_params_vec.back().second;
  REQUIRE(next_dep[0] == 0u); /* neuron index */
  REQUIRE(next_dep[1] == 1u); /* input_synapse_index */
  REQUIRE(next_dep[2] == 0u); /* weight_synapse_index */
  REQUIRE(next_dep[3] == 0u); /* start_inside_input_synapse */
  REQUIRE(next_dep[4] ==
          synapse_size / 3u + 1u); /* start_inside_weight_synapse */

  /* Construct the second part of the synapses */
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep});

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec2 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec2.size());
  REQUIRE(dep_params_vec2.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  /* The one after the next dependency should point to the last part of the
   * Neuron input before a bias operation */
  const auto &next_dep2 = dep_params_vec2.back().second;
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep2});
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec3 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec3.size());
  REQUIRE(dep_params_vec3.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep3 = dep_params_vec3.back().second;
  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep3});

  /* The next dependency should point to the next part of the Neuron input */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec4 = dependency_requests->first;
  REQUIRE(1u == dep_params_vec4.size());
  REQUIRE(dep_params_vec4.back().first ==
          rafko_gym::ad_operation_neuron_bias_d);

  const auto &next_dep4 = dep_params_vec3.back().second;
  REQUIRE(next_dep4[0] == 0u);                /* neuron index */
  REQUIRE(next_dep4[1] == synapse_size + 1u); /* neuron_weight_index */
  /*!Note: Bias operation takes the weight synapse index, in this case it's
   * spike weight + inputs + bias */
}

TEST_CASE("Testing if Neuron Input operation throws an error on construction "
          "if staring synapse indexes do not match",
          "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(-1);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);
  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size + 2); /* spike weight + inputs + 1 bias weight*/

  /* Construct Neuron Input operation and make it request dependencies */
  rafko_gym::RafkoBackpropagationData data(network);
  REQUIRE_THROWS(rafko_gym::RafkoBackpropNeuronInputOperation(
      data, network, 0u /*operation_index*/, 0 /*neuron_index*/,
      0 /*input_synapse_index*/, 0 /*weight_synapse_index*/,
      5 /*start_inside_input_synapse*/, 1 /*start_inside_weight_synapse*/));
}

/*==============================================================================
 * Testing neuron data dependencies
 *==============================================================================
 **/
TEST_CASE("Testing if Neuron input dependencies are generating correct "
          "dependency requests for a contigous whole synapse; For internal "
          "neuron inputs only",
          "[optimizer][back-propagation][neuron-input]") {
  /* Construct one single neuron with a simple contigous synapse */
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(0);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);
  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size + 1); /* spike weight + inputs */

  /* Construct Neuron Input operation and make it request dependencies */
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  /* Required dependencies should contain the first 10 neuron index values */
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(synapse_size == dep_params_vec.size());

  std::set<std::uint32_t> neuron_index_values_left_out = {0, 1, 2, 3, 4,
                                                          5, 6, 7, 8, 9};
  for (const auto &[operation_type, operation_init_vec] : dep_params_vec) {
    if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
      neuron_index_values_left_out.erase(operation_init_vec[0]);
    }
  }
  REQUIRE(neuron_index_values_left_out.empty());
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where input synapse is cut in the middle; For "
    "internal inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(0);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 2u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(synapse_size / 2);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 2u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size + 2u); /* spike weight + inputs + 1 bias weight*/

  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);

  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(synapse_size / 2u + 1u ==
          dep_params_vec.size()); /* Inputs + next operation */
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep = dep_params_vec.back().second;
  std::set<std::uint32_t> neuron_index_values_left_out = {0, 1, 2, 3, 4,
                                                          5, 6, 7, 8, 9};
  for (const auto &[operation_type, operation_init_vec] : dep_params_vec) {
    if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
      neuron_index_values_left_out.erase(operation_init_vec[0]);
    }
  }

  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep});

  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec2 = dependency_requests->first;
  REQUIRE(synapse_size / 2u + 1 ==
          dep_params_vec2.size()); /* Inputs + next dependency */
  REQUIRE(dep_params_vec2.back().first ==
          rafko_gym::ad_operation_neuron_bias_d);

  for (const auto &[operation_type, operation_init_vec] : dep_params_vec2) {
    if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
      neuron_index_values_left_out.erase(operation_init_vec[0]);
    }
  }

  REQUIRE(neuron_index_values_left_out.empty());
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where weight synapse is cut in the middle; For "
    "internal inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 10u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(0);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u + 1u); /* spike weight + half of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 2u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 2u + 1u); /* bias weight + half of the inputs */

  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);
  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec = dependency_requests->first;
  REQUIRE(synapse_size / 2u + 1u ==
          dep_params_vec.size()); /* Inputs + next operation */
  REQUIRE(dep_params_vec.back().first ==
          rafko_gym::ad_operation_neuron_input_d);

  const auto &next_dep = dep_params_vec.back().second;
  std::set<std::uint32_t> neuron_index_values_left_out = {0, 1, 2, 3, 4,
                                                          5, 6, 7, 8, 9};
  for (const auto &[operation_type, operation_init_vec] : dep_params_vec) {
    if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
      neuron_index_values_left_out.erase(operation_init_vec[0]);
    }
  }

  dependency_requests = call_neuron_input_dep_request(
      network, 0u /*operation_index*/, {next_dep});

  REQUIRE(dependency_requests.has_value());
  const DependencyParams &dep_params_vec2 = dependency_requests->first;
  REQUIRE(synapse_size / 2u + 1 ==
          dep_params_vec2.size()); /* Inputs + next dependency */
  REQUIRE(dep_params_vec2.back().first ==
          rafko_gym::ad_operation_neuron_bias_d);

  for (const auto &[operation_type, operation_init_vec] : dep_params_vec2) {
    if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
      neuron_index_values_left_out.erase(operation_init_vec[0]);
    }
  }

  REQUIRE(neuron_index_values_left_out.empty());
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where weight synapse is cut in the middle and "
    "the input synapse is cut two times; For internal inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 12u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(0);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(synapse_size / 3);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(2)->set_starts(2 * synapse_size / 3);
  neuron.mutable_input_indices(2)->set_interval_size(synapse_size / 3u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u + 1u); /* spike weight + half of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 2u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 2u + 1u); /* half of the inputs + bias weight */

  std::set<std::uint32_t> neuron_index_values_left_out = {0, 1, 2, 3, 4,  5,
                                                          6, 7, 8, 9, 10, 11};
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);
  std::uint32_t spike_count = 0u;
  for (std::uint32_t part = 0; part < 4; ++part) {
    REQUIRE(dependency_requests.has_value());
    const DependencyParams &dep_params_vec = dependency_requests->first;
    const auto &next_dep = dep_params_vec.back().second;
    for (const auto &[operation_type, operation_init_vec] : dep_params_vec) {
      if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
        ++spike_count;
        neuron_index_values_left_out.erase(operation_init_vec[0]);
      }
    }
    if (dep_params_vec.back().first != rafko_gym::ad_operation_neuron_bias_d) {
      dependency_requests = call_neuron_input_dep_request(
          network, 0u /*operation_index*/, {next_dep});
    }
  }
  REQUIRE(spike_count == synapse_size);
  REQUIRE(neuron_index_values_left_out.empty());
}

TEST_CASE(
    "Testing if Neuron input dependencies are generating correct dependency "
    "requests for two synapses, where weight synapse is cut in two times and "
    "the input synapse is cut in the middle; For internal inputs only",
    "[optimizer][back-propagation][neuron-input]") {
  constexpr std::uint32_t synapse_size = 12u;
  rafko_net::RafkoNet network;
  rafko_net::Neuron example_neuron;
  *network.add_neuron_array() = example_neuron;
  rafko_net::Neuron &neuron = *network.mutable_neuron_array(0u);

  neuron.add_input_indices();
  neuron.mutable_input_indices(0)->set_starts(0);
  neuron.mutable_input_indices(0)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(1)->set_starts(synapse_size / 3);
  neuron.mutable_input_indices(1)->set_interval_size(synapse_size / 3u);
  neuron.add_input_indices();
  neuron.mutable_input_indices(2)->set_starts(2 * synapse_size / 3);
  neuron.mutable_input_indices(2)->set_interval_size(synapse_size / 3u);

  neuron.add_input_weights();
  neuron.mutable_input_weights(0)->set_starts(0);
  neuron.mutable_input_weights(0)->set_interval_size(
      synapse_size / 2u + 1u); /* spike weight + half of the inputs */
  neuron.add_input_weights();
  neuron.mutable_input_weights(1)->set_starts(synapse_size / 2u + 1u);
  neuron.mutable_input_weights(1)->set_interval_size(
      synapse_size / 2u + 1u); /* half of the inputs + bias weight */

  std::set<std::uint32_t> neuron_index_values_left_out = {0, 1, 2, 3, 4,  5,
                                                          6, 7, 8, 9, 10, 11};
  auto dependency_requests =
      call_neuron_input_dep_request(network, 0u /*operation_index*/);
  std::uint32_t spike_count = 0u;
  for (std::uint32_t part = 0; part < 4; ++part) {
    REQUIRE(dependency_requests.has_value());
    const DependencyParams &dep_params_vec = dependency_requests->first;
    const auto &next_dep = dep_params_vec.back().second;
    for (const auto &[operation_type, operation_init_vec] : dep_params_vec) {
      if (operation_type == rafko_gym::ad_operation_neuron_spike_d) {
        ++spike_count;
        neuron_index_values_left_out.erase(operation_init_vec[0]);
      }
    }
    if (dep_params_vec.back().first != rafko_gym::ad_operation_neuron_bias_d) {
      dependency_requests = call_neuron_input_dep_request(
          network, 0u /*operation_index*/, {next_dep});
    }
  }
  REQUIRE(spike_count == synapse_size);
  REQUIRE(neuron_index_values_left_out.empty());
}

} /* namespace rafko_net_test */
