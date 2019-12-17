#include <random>

#include "catch.hpp"
#include "test_mockups.h"

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
using sparse_net_library::TRANSFER_FUNC_IDENTITY;
using sparse_net_library::Partial_solution;
using sparse_net_library::Partial_solution_solver;
using sparse_net_library::Transfer_function_info;

/*###############################################################################################
 * Testing if the solver processes a partial_solution detail correctly
 * - Construct a partial_solution detail
 *   - 2 inputs
 *   - 2 neurons
 *     - The first neuron has the inputs and the second has the first neuron
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
sdouble32 neuron1_result = (
  (network_inputs[0] * partial_solution->weight_table(0)) + (network_inputs[1] * partial_solution->weight_table(1))
  + partial_solution->biases(0)
);
Transfer_function_info::apply_to_data(partial_solution->neuron_transfer_functions(0),neuron1_result);

sdouble32 neuron2_result = (neuron1_result * partial_solution->weight_table(2)) + partial_solution->biases(1);
Transfer_function_info::apply_to_data(partial_solution->neuron_transfer_functions(1),neuron2_result);

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
  partial_solution.add_actual_index(0u); /* Really doesn't matter that much in this testcase */
  partial_solution.add_actual_index(1u); /* It will matter only when multiple partial partial_solutions are joind together */

  /**###################################################################################################
   * The first neuron shall have the inputs 
   */
  partial_solution.add_input_sizes(2u); 
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNC_IDENTITY);
  partial_solution.add_memory_ratios(0.0);
  partial_solution.add_biases(0.0);
  partial_solution.add_inside_indexes(0); /* input 1 goes to neuron1 */
  partial_solution.add_weight_indexes(0);
  partial_solution.add_inside_indexes(1); /* input2 goes to neuron2 */
  partial_solution.add_weight_indexes(1);

  /**###################################################################################################
   * The second Neuron shall have the first neuron as input
   */
  partial_solution.add_input_sizes(1u);
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNC_IDENTITY);
  partial_solution.add_memory_ratios(0.0);
  partial_solution.add_biases(0.0);
  partial_solution.add_inside_indexes(2); /* neuron1 goes to neuron2;  that is the output which isn't in the inside indexes */
  partial_solution.add_weight_indexes(2);

  /* Run the partial_solution */
  Partial_solution_solver solver;

  /* The result should be according to the calculations */
  CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

  /* The result should change in accordance with the parameters */
  srand (time(nullptr));
  for(uint8 variant_iterator = 0; variant_iterator < 30; variant_iterator++){
    partial_solution.set_weight_table(0,static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(1,static_cast<sdouble32>(rand()%11) / 10.0);
    partial_solution.set_weight_table(0,static_cast<sdouble32>(rand()%11) / 10.0);
    CHECK( solver.solve(&partial_solution,&network_inputs)[0] == partial_solution_result(network_inputs, &partial_solution) );

  }

  /*!#8 Todo: biases, memory ratios, transfer funtions */

  /* Different memory ratios should bring different results */
}

} /* namespace sparse_net_library_test */
