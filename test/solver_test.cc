#include "catch.hpp"
#include "test_mockups.h"

#include "sparsenet_global.h"
#include "models/sNet.pb.h"
#include "services/sparse_net_solver.h"

namespace sparse_net_library_test {

	using sparse_net_library::sdouble32;
	using sparse_net_library::uint16;
	using sparse_net_library::uint32;
  using sparse_net_library::transfer_functions;
  using sparse_net_library::TRANSFER_FUNC_IDENTITY;

  class Solver_mockup : sparse_net_library::SparseNetSolver{
  public:
    struct Solution_detail : public sparse_net_library::SparseNetSolver::Solution_detail { };
  };

  /*###############################################################################################
   * Testing if the solver processes a solution detail correctly
   * - Construct a solution detail
   *   - 2 inputs
   *   - 2 neurons
   *     - The first neuron has the inputs and the second has the first neuron
   * - See if it is solved correctly with multiple variations
   *   - different input numbers
   *   - different weights
   *   - different biases
   */
TEST_CASE( "Solving an artificial solution detail", "[solve][small][manual]" ){
  /* Prepare a solution */
  Solver_mockup::Solution_detail fabricated_solution;
  fabricated_solution.internal_neuron_number = 2;
  fabricated_solution.input_data_size = 2;
  fabricated_solution.data = {0.0,0.0,0.0,0.0};
  fabricated_solution.actual_index = {0u,1u};
  fabricated_solution.input_sizes = {2u,1u};
  fabricated_solution.transfer_functions =   {TRANSFER_FUNC_IDENTITY,TRANSFER_FUNC_IDENTITY};
  fabricated_solution.memory_ratios = {0.0,0.0};
  fabricated_solution.biases = {0.0,0.0};
  fabricated_solution.inside_indexes = {0u,1u,2u};
  fabricated_solution.weights = {1.0,1.0,1.0};

  CHECK( true );
}

} /* namespace sparse_net_library_test */
