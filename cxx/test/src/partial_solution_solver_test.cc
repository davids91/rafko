#include <random>

#include "test/catch.hpp"
#include "test/test_mockups.h"

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/transfer_function_info.h"
#include "services/partial_solution_solver.h"
#include "services/sparse_net_solver.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library_test {

using std::vector;

using sparse_net_library::sdouble32;
using sparse_net_library::uint8;
using sparse_net_library::uint16;
using sparse_net_library::uint32;
using sparse_net_library::transfer_functions;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::Partial_solution;
using sparse_net_library::Partial_solution_solver;
using sparse_net_library::Transfer_function_info;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Synapse_interval;

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
 *//*!#8 */

TEST_CASE( "Solving an artificial partial_solution detail", "[solve][partial_solution][manual]" ){
  Partial_solution partial_solution;
  vector<uint32> helper_vector_uint;
  vector<sdouble32> neuron_output;
  vector<sdouble32> expected_neuron_output;
  Synapse_interval temp_synapse_interval;

  /* Define the input to the network */
  vector<sdouble32> network_inputs = {10.0,5.0};
  vector<sdouble32> collected_inputs;

  /* Prepare a partial_solution */
  manual_2_neuron_partial_solution(partial_solution, network_inputs.size());
  
  /* Add relevant Partial solution input (the input of the first @Neuron) */
  temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(0));
  temp_synapse_interval.set_interval_size(network_inputs.size());
  *partial_solution.add_input_data() = temp_synapse_interval;
  
  /* Test the partial_solution */
  Partial_solution_solver solver(partial_solution);
  solver.collect_input_data(network_inputs,{});
  REQUIRE( network_inputs.size() == solver.get_input_size() );

  /* The result should be according to the calculations */
  neuron_output = solver.solve();
  expected_neuron_output = vector<sdouble32>(2);
  manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
  CHECK( Approx(neuron_output[1]).epsilon(0.00000000000001) == expected_neuron_output[1] );

  /* The result should change in accordance with the parameters */
  srand (time(nullptr));
  for(uint8 variant_iterator = 0; variant_iterator < 100; variant_iterator++){
    for(uint16 i; i <= network_inputs.size(); ++i){ /* Set weight s for the first 2 neurons = input weights + the first Neuron Weight */
      partial_solution.set_weight_table(i,static_cast<sdouble32>(rand()%11) / 10.0);
    }

    neuron_output = solver.solve();
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_output[1]).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_weight_table(partial_solution.bias_index(0),static_cast<sdouble32>(rand()%110) / 10.0);
    partial_solution.set_weight_table(partial_solution.bias_index(1),static_cast<sdouble32>(rand()%110) / 10.0);
    neuron_output = solver.solve();
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_output[1]).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_weight_table(partial_solution.memory_ratio_index(0),static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(partial_solution.memory_ratio_index(1),static_cast<sdouble32>(rand()%11) / 10.0);
    neuron_output = solver.solve();
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_output[1]).epsilon(0.00000000000001) == expected_neuron_output[1] );

    partial_solution.set_neuron_transfer_functions(rand()%(partial_solution.neuron_transfer_functions_size()),Transfer_function_info::next());
    neuron_output = solver.solve();
    manual_2_neuron_result(network_inputs, expected_neuron_output, partial_solution);
    CHECK( Approx(neuron_output[1]).epsilon(0.00000000000001) == expected_neuron_output[1] );
  }
}

/*###############################################################################################
 * Testing if the partial solution solver collects its relevant input correctly
 * - define a 10 element input array
 * - define different partition ranges based on it
 * - define the partial solution so every neuon gives back the corresponding input
 * - see if the input is collected correctly
 */
TEST_CASE("Test Partial solution input collection","[solve][partial_solution][input_collection]"){
  Partial_solution partial_solution;
  vector<sdouble32> network_inputs = {1.9,2.8,3.7,4.6,5.5,6.4,7.3,8.2,9.1,10.0};
  vector<sdouble32> collected_inputs;
  Synapse_interval temp_synapse_interval;

  partial_solution.set_internal_neuron_number(network_inputs.size());
  partial_solution.add_weight_table(0.0);  /* A weight for the biases and memory ratios */
  for(uint32 i = 0; i < network_inputs.size(); ++i){
    partial_solution.add_weight_table(1.0); 
    partial_solution.add_actual_index(i);
    partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
    partial_solution.add_memory_ratio_index(0);
    partial_solution.add_bias_index(0);

    partial_solution.add_index_synapse_number(1); /* 1 synapse for indexes and 1 for weights */
    temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(i));
    temp_synapse_interval.set_interval_size(1u); /* Input index synapse starts at the beginning of the data and goes on for an interval of 1 */
    *partial_solution.add_inside_indices() = temp_synapse_interval;

    partial_solution.add_weight_synapse_number(1);
    temp_synapse_interval.set_starts(1u);
    temp_synapse_interval.set_interval_size(1u); /* weight of 1 here */
    *partial_solution.add_weight_indices() = temp_synapse_interval;
  } 

  /**###################################################################################################
   * Add the partial solution inputs
   */
  /* First 3 elements */
  temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(0));
  temp_synapse_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Elements from 3 to 5 */
  temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(3));
  temp_synapse_interval.set_interval_size(3);
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Elements from 6 to 8 */
  temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(6));
  temp_synapse_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Elements from 8 to 9 ( to the end ) */
    temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(8));
  temp_synapse_interval.set_interval_size(2);
  *partial_solution.add_input_data() = temp_synapse_interval;

  /* Prepare the partial solution */
  Partial_solution_solver solver(partial_solution);

  REQUIRE( network_inputs.size() == solver.get_input_size() );

  solver.collect_input_data(network_inputs,{});
  collected_inputs = solver.solve();
  for(uint32 i = 0; i < network_inputs.size(); ++i){
    REQUIRE( network_inputs[i] == collected_inputs[i]);
  }
}

} /* namespace sparse_net_library_test */
