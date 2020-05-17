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

#include <vector>
#include <memory>

#include "test/test_utility.h"

#include "gen/solution.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "services/synapse_iterator.h"
#include "services/sparse_net_builder.h"
#include "services/solution_builder.h"
#include "services/weight_updater.h"

namespace sparse_net_library_test {

using std::unique_ptr;
using std::make_unique;

using sparse_net_library::uint8;
using sparse_net_library::uint16;
using sparse_net_library::uint32;
using sparse_net_library::sint32;
using sparse_net_library::sdouble32;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::Solution_builder;
using sparse_net_library::SparseNet;
using sparse_net_library::Partial_solution;
using sparse_net_library::Solution;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Weight_updater;

/*###############################################################################################
 * Testing if the weight updater are updating a generated solution correctly
 * - Create a network, solution and weight updater
 * - update the weights of the network
 * - Check if the updated weights match the ones copied to the solution
 */
TEST_CASE("Weight updater test","[build][weight-update]"){
  vector<uint32> net_structure = {2,4,3,1,2};
  vector<sdouble32> net_input = {double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)};

  /* Build the above described net, solution and a Weight updater */
  unique_ptr<SparseNet> net(
    Sparse_net_builder().input_size(5).expected_input_range(double_literal(5.0))
    .cost_function(COST_FUNCTION_MSE).dense_layers(net_structure)
  );
  Weight_updater weight_updater(*net);
  Solution solution = *Solution_builder().service_context().build(*net);
  check_if_the_same(*net, solution);

  /* Change the weights in the network */
  srand (time(nullptr));
  for(sint32 weight_iterator = 0; weight_iterator < net->weight_table_size() ; ++weight_iterator){
    net->set_weight_table(weight_iterator,(static_cast<sdouble32>(rand()%11) / double_literal(10.0)));
  }

  /* Copy the weights into the Solution */
  weight_updater.update_solution_with_weights(solution);
  
  check_if_the_same(*net, solution);
}

} /* namespace sparse_net_library_test */
