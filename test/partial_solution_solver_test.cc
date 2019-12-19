#include <random>

#include "test/catch.hpp"
#include "test/test_mockups.h"

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"
#include "models/gen/solution.pb.h"
#include "models/transfer_function_info.h"
#include "services/partial_solution_solver.h"
#include "services/sparse_net_solver.h"

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
/** @brief Calculates the result of the partial partial_solution manually based on the given inputs,
* in case the structure of the partial partial_solution equals with the one described above in the testcase
*/
sdouble32 partial_solution_result(vector<sdouble32> network_inputs, const Partial_solution * partial_solution){
  sdouble32 neuron1_result = ( /* Neuron 1 = transfer_function( ( input0 * weight0 + input1 * weight1 ) + bias0 )*/
    (network_inputs[0] * partial_solution->weight_table(0)) + (network_inputs[1] * partial_solution->weight_table(1))
    + partial_solution->weight_table(partial_solution->bias_index(0))
  );
  Transfer_function_info::apply_to_data(partial_solution->neuron_transfer_functions(0),neuron1_result);
  neuron1_result *= (1.0 - partial_solution->weight_table(partial_solution->memory_ratio_index(0)));

  /* Neuron 2 = transfer_function( (Neuron1 * weight2) + bias1 ) */
  sdouble32 neuron2_result = (neuron1_result * partial_solution->weight_table(2))
   + partial_solution->weight_table(partial_solution->bias_index(1));
  Transfer_function_info::apply_to_data(partial_solution->neuron_transfer_functions(1),neuron2_result);
  neuron2_result *= (1.0 - partial_solution->weight_table(partial_solution->memory_ratio_index(1)));

  return neuron2_result;
}

TEST_CASE( "Solving an artificial partial_solution detail", "[solve][small][manual]" ){
  Partial_solution partial_solution;
  vector<uint32> helper_vector_uint;
  vector<sdouble32> helper_vector_double;

  /* Define the input to the network */
  vector<sdouble32> network_inputs = {10.0,5.0};

  /* Prepare a partial_solution */
  partial_solution.set_internal_neuron_number(2);
  partial_solution.set_input_data_size(2);
  partial_solution.add_weight_table(1.0); /* Every 3 weights shall be modified in this example, so they'll all have thir own weight table entry */
  partial_solution.add_weight_table(1.0);
  partial_solution.add_weight_table(1.0);
  partial_solution.add_weight_table(0.0); /* Memory ratios are also stored here */
  partial_solution.add_weight_table(0.0);
  partial_solution.add_weight_table(50.0); /* Biases are also stored here */
  partial_solution.add_weight_table(10.0);
  partial_solution.add_actual_index(0u); /* Really doesn't matter that much in this testcase */
  partial_solution.add_actual_index(1u); /* It will matter only when multiple partial partial_solutions are joind together */

  /**###################################################################################################
   * The first neuron shall have the inputs 
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_ratio_index(3);
  partial_solution.add_bias_index(5);
  partial_solution.add_index_synapse_number(1); /* 1 synapse for indexes and 1 for weights*/
  partial_solution.add_weight_synapse_number(1);

  /* input 1 and 2 goes to neuron1 */
  partial_solution.add_inside_index_sizes(2); /* Neuron 1 has an input index synapse of 2 indexes ( first 2 inputs ) */
  partial_solution.add_inside_index_starts(0); /* Input index synapse starts at the beginning of the data */
  partial_solution.add_weight_index_sizes(2); /* Neuron 1 has the first two weights in its only weight synapse */
  partial_solution.add_weight_index_starts(0);

  /**###################################################################################################
   * The second Neuron shall have the first neuron as input
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_ratio_index(4);
  partial_solution.add_bias_index(6);
  partial_solution.add_index_synapse_number(1); /* 1 synapse for indexes and 1 for weights*/
  partial_solution.add_weight_synapse_number(1);

  /* neuron1 goes to neuron2;  that is the output which isn't in the inside indexes */
  partial_solution.add_inside_index_sizes(1); /* Neuron 2 has an input synapse of size 1*/
  partial_solution.add_inside_index_starts(partial_solution.input_data_size()); /* The input synapse starts at the 2nd index of the data array */
  partial_solution.add_weight_index_sizes(1); /* Neuron 2 has a an weight synapse of size 1 */
  partial_solution.add_weight_index_starts(2); /* The weight synapse starts at index 2 of the weight table */

  /* Run the partial_solution */
  Partial_solution_solver solver;

  /* The result should be according to the calculations */
  CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

  /* The result should change in accordance with the parameters */
  srand (time(nullptr));
  for(uint8 variant_iterator = 0; variant_iterator < 100; variant_iterator++){
    partial_solution.set_weight_table(0,static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(1,static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(2,static_cast<sdouble32>(rand()%11) / 10.0);
    CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

    partial_solution.set_weight_table(partial_solution.bias_index(0),static_cast<sdouble32>(rand()%110) / 10.0);
    partial_solution.set_weight_table(partial_solution.bias_index(1),static_cast<sdouble32>(rand()%110) / 10.0);
    CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

    partial_solution.set_weight_table(partial_solution.memory_ratio_index(0),static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(partial_solution.memory_ratio_index(1),static_cast<sdouble32>(rand()%11) / 10.0);
    CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

    partial_solution.set_neuron_transfer_functions(rand()%(partial_solution.neuron_transfer_functions_size()),Transfer_function_info::next());
    CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );
  }

  /*!#8 */
}

} /* namespace sparse_net_library_test */
