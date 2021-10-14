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
#include "test/catch.hpp"
#include "test/test_utility.h"

#include <random>

#include "rafko_protocol/sparse_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "rafko_net/models/transfer_function.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_net/services/partial_solution_solver.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net_test {

using std::vector;
using std::reference_wrapper;

using rafko_net::transfer_functions;
using rafko_net::TRANSFER_FUNCTION_IDENTITY;
using rafko_utilities::DataRingbuffer;
using rafko_net::Partial_solution;
using rafko_net::Partial_solution_solver;
using rafko_net::Transfer_function;
using rafko_net::Index_synapse_interval;
using rafko_net::Input_synapse_interval;
using rafko_net::Synapse_iterator;
using rafko_mainframe::Service_context;

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
  Service_context service_context;
  DataRingbuffer neuron_data(1,2);
  Partial_solution partial_solution;
  vector<uint32> helper_vector_uint;
  vector<sdouble32> expected_neuron_output;
  Input_synapse_interval temp_synapse_interval;

  /* Define the input and structure of the network */
  vector<sdouble32> network_inputs = {double_literal(10.0),double_literal(5.0)};
  manual_2_neuron_partial_solution(partial_solution, network_inputs.size());

  /* Add relevant Partial solution input (the input of the first @Neuron) */
  temp_synapse_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0));
  temp_synapse_interval.set_interval_size(network_inputs.size());
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Test the partial_solution */
  Partial_solution_solver solver(partial_solution, service_context);

  /* The result should be according to the calculations */
  solver.solve(network_inputs, neuron_data);
  expected_neuron_output = vector<sdouble32>(2);
  manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
  CHECK( Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

  /* The result should change in accordance with the parameters */
  srand (time(nullptr));
  for(uint8 variant_iterator = 0; variant_iterator < 100; variant_iterator++){
    Synapse_iterator<>::iterate(partial_solution.weight_indices(),[&](Index_synapse_interval weight_synapse, sint32 neuron_weight_index){
      partial_solution.set_weight_table(neuron_weight_index,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
    },0u,1u); /* Mess with the weights of the first Neuron */
    Synapse_iterator<>::iterate(partial_solution.weight_indices(),[&](Index_synapse_interval weight_synapse, sint32 neuron_weight_index){
      partial_solution.set_weight_table(neuron_weight_index,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
    },1u,1u); /* Mess with the weights of the second Neuron */

    solver.solve(network_inputs, neuron_data);
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

    solver.solve(network_inputs, neuron_data);
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_weight_table(partial_solution.memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
    partial_solution.set_weight_table(partial_solution.memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
    solver.solve(network_inputs, neuron_data);
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_neuron_transfer_functions(rand()%(partial_solution.neuron_transfer_functions_size()),Transfer_function::next());
    solver.solve(network_inputs, neuron_data);
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    REQUIRE( Approx(neuron_data.get_element(0,1)).epsilon(0.00000000000001) == expected_neuron_output[1] );
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
  Service_context service_context;
  Partial_solution partial_solution;
  vector<sdouble32> network_inputs = {double_literal(1.9),double_literal(2.8),double_literal(3.7),double_literal(4.6),double_literal(5.5),double_literal(6.4),double_literal(7.3),double_literal(8.2),double_literal(9.1),double_literal(10.0)};
  Index_synapse_interval temp_index_interval;
  Input_synapse_interval temp_input_interval;
  DataRingbuffer neuron_data(1, network_inputs.size());

  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(network_inputs.size());
  *partial_solution.mutable_output_data() = temp_index_interval;
  partial_solution.add_weight_table(double_literal(0.0));  /* A weight for the memory filter */
  for(uint32 i = 0; i < network_inputs.size(); ++i){
    partial_solution.add_weight_table(double_literal(1.0));
    partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
    partial_solution.add_memory_filter_index(0);

    partial_solution.add_index_synapse_number(1); /* 1 synapse for indexes and 1 for weights */
    temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(i));
    temp_input_interval.set_interval_size(1u); /* Input index synapse starts at the beginning of the data and goes on for an interval of 1 */
    *partial_solution.add_inside_indices() = temp_input_interval;

    partial_solution.add_weight_synapse_number(1);
    temp_index_interval.set_starts(1u);
    temp_index_interval.set_interval_size(1u); /* weight of 1 here */
    *partial_solution.add_weight_indices() = temp_index_interval;
  }

  /**###################################################################################################
   * Add the partial solution inputs
   */
  /* First 3 elements */
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0));
  temp_input_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 3 to 5 */
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(3));
  temp_input_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 6 to 8 */
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(6));
  temp_input_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Elements from 8 to 9 ( to the end ) */
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(8));
  temp_input_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_input_interval;

  /* Prepare the partial solution */
  Partial_solution_solver solver(partial_solution, service_context);

  solver.solve(network_inputs, neuron_data); /* Since the network just spits the inputs back out so the input collection is testable through it*/
  for(uint32 i = 0; i < network_inputs.size(); ++i){
    REQUIRE( Approx(network_inputs[i]).epsilon(0.00000000000001) == neuron_data.get_element(0,i));
  }
}

} /* namespace rafko_net_test */
