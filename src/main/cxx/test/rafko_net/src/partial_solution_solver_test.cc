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

#include <random>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#include "rafko_net/services/partial_solution_solver.hpp"
#include "rafko_net/services/synapse_iterator.hpp"

#include "test/test_utility.hpp"

namespace rafko_net_test {

/*###############################################################################################
 * Testing if the solver processes a partial_solution detail correctly
 * - Construct a partial_solution detail
 *   - 2 inputs
 *   - 2 Neurons: The first neuron has the inputs and the second has the first neuron
 *   - The end result should be : input1 * weight
 * - See if it is solved correctly with multiple variations
 *   - different input numbers
 *   - different weights
 *   - different biases
 */

TEST_CASE( "Solving an artificial partial_solution detail", "[solve][partial-solution][manual]" ){
  rafko_mainframe::RafkoSettings settings;
  rafko_utilities::DataRingbuffer<> neuron_data(1u, [](std::vector<double>& element){
    element.resize(2u);
  });
  rafko_net::PartialSolution partial_solution;
  std::vector<std::uint32_t> helper_vector_uint;
  std::vector<double> expected_neuron_output;
  rafko_net::InputSynapseInterval temp_synapse_interval;

  /* Define the input and structure of the network */
  std::vector<double> network_inputs = {(10.0),(5.0)};
  rafko_test::manual_2_neuron_partial_solution(partial_solution, network_inputs.size());

  /* Add relevant Partial solution input (the input of the first @Neuron) */
  temp_synapse_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(0));
  temp_synapse_interval.set_interval_size(network_inputs.size());
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Test the partial_solution */
  rafko_net::PartialSolutionSolver solver(partial_solution, settings);

  /* The result should be according to the calculations */
  solver.solve(network_inputs, neuron_data);
  expected_neuron_output = std::vector<double>(2);
  rafko_test::manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
  CHECK( Catch::Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

  /* The result should change in accordance with the parameters */
  srand (time(nullptr));
  for(std::uint8_t variant_iterator = 0; variant_iterator < 100u; variant_iterator++){
    rafko_net::SynapseIterator<>::iterate(partial_solution.weight_indices(),[&](std::int32_t neuron_weight_index){
      partial_solution.set_weight_table(neuron_weight_index,static_cast<double>(rand()%11) / (10.0));
    },0u,1u); /* Mess with the weights of the first Neuron */
    rafko_net::SynapseIterator<>::iterate(partial_solution.weight_indices(),[&](std::int32_t neuron_weight_index){
      partial_solution.set_weight_table(neuron_weight_index,static_cast<double>(rand()%11) / (10.0));
    },1u,1u); /* Mess with the weights of the second Neuron */

    solver.solve(network_inputs, neuron_data);
    rafko_test::manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Catch::Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

    solver.solve(network_inputs, neuron_data);
    rafko_test::manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Catch::Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_neuron_transfer_functions(rand()%(partial_solution.neuron_transfer_functions_size()),rafko_net::TransferFunction::next());
    solver.solve(network_inputs, neuron_data);
    rafko_test::manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    REQUIRE( Catch::Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );
  }
}

/*###############################################################################################
 * Testing if the partial solution solver collects its relevant input correctly
 * - define a 10 element input array
 * - define different partition ranges based on it
 * - define the partial solution so every neuon gives back the corresponding input
 * - see if the input is collected correctly
 */
TEST_CASE("Test Partial solution input collection","[solve][partial-solution][input_collection]"){
  rafko_mainframe::RafkoSettings settings;
  rafko_net::PartialSolution partial_solution;
  std::vector<double> network_inputs = {(1.9),(2.8),(3.7),(4.6),(5.5),(6.4),(7.3),(8.2),(9.1),(10.0)};
  rafko_net::IndexSynapseInterval temp_index_interval;
  rafko_net::InputSynapseInterval temp_input_interval;
  rafko_utilities::DataRingbuffer<> neuron_data(1u,[&network_inputs](std::vector<double>& element){
    element.resize(network_inputs.size());
  });

  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(network_inputs.size());
  *partial_solution.mutable_output_data() = temp_index_interval;
  partial_solution.add_weight_table((0.0));  /* A weight for the spike function */
  for(std::uint32_t i = 0; i < network_inputs.size(); ++i){
    partial_solution.add_weight_table((1.0));
    partial_solution.add_neuron_input_functions(rafko_net::input_function_add);
    partial_solution.add_neuron_transfer_functions(rafko_net::transfer_function_identity);
    partial_solution.add_neuron_spike_functions(rafko_net::spike_function_memory);

    partial_solution.add_index_synapse_number(1); /* 1 synapse for indexes and 1 for weights */
    temp_input_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(i));
    temp_input_interval.set_interval_size(1u); /* Input index synapse starts at the beginning of the data and goes on for an interval of 1 */
    *partial_solution.add_inside_indices() = temp_input_interval;

    partial_solution.add_weight_synapse_number(1);
    temp_index_interval.set_starts(0u);
    temp_index_interval.set_interval_size(1u + 1u); /* weight1 + 1 weight for the spike function*/
    *partial_solution.add_weight_indices() = temp_index_interval;
  }

  /**###################################################################################################
   * Add the partial solution inputs
   */
  /* First 3 elements */
  temp_input_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(0));
  temp_input_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 3 to 5 */
  temp_input_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(3));
  temp_input_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 6 to 8 */
  temp_input_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(6));
  temp_input_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 8 to 9 ( to the end ) */
  temp_input_interval.set_starts(rafko_net::SynapseIterator<>::external_index_from_array_index(8));
  temp_input_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Prepare the partial solution */
  rafko_net::PartialSolutionSolver solver(partial_solution, settings);

  solver.solve(network_inputs, neuron_data); /* Since the network just spits the inputs back out so the input collection is testable through it*/
  for(std::uint32_t i = 0; i < network_inputs.size(); ++i){
    REQUIRE( Catch::Approx(network_inputs[i]).epsilon(0.00000000000001) == neuron_data.get_element(0,i));
  }
}

} /* namespace rafko_net_test */
