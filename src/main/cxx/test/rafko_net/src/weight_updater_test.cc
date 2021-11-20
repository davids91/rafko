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

#include <vector>
#include <memory>

#include "rafko_protocol/solution.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/rafko_weight_updater.h"

namespace rafko_net_test {

using std::unique_ptr;
using std::make_unique;

using rafko_net::RafkoNetBuilder;
using rafko_net::SolutionBuilder;
using rafko_net::RafkoNet;
using rafko_net::PartialSolution;
using rafko_net::Solution;
using rafko_net::SynapseIterator;
using rafko_net::RafkoWeightUpdater;
using rafko_mainframe::RafkoServiceContext;

/*###############################################################################################
 * Testing if the weight updater are updating a generated solution correctly
 * - Create a network, solution and weight updater
 * - update the weights of the network
 * - Check if the updated weights match the ones copied to the solution
 */
TEST_CASE("Weight updater test","[build][weight-update]"){
  RafkoServiceContext service_context;
  vector<uint32> net_structure = {2,4,3,1,2};
  vector<sdouble32> net_input = {double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)};
  unique_ptr<RafkoNet> net(RafkoNetBuilder(service_context).input_size(5).expected_input_range(double_literal(5.0)).dense_layers(net_structure));
  unique_ptr<Solution> solution = unique_ptr<Solution>(SolutionBuilder(service_context).build(*net));
  RafkoWeightUpdater weight_updater(*net, *solution, service_context);
  check_if_the_same(*net, *solution);

  /* Change the weights in the network and take them over into the generated solution */
  srand (time(nullptr));
  for(sint32 weight_iterator = 0; weight_iterator < net->weight_table_size() ; ++weight_iterator){
    net->set_weight_table(weight_iterator,(static_cast<sdouble32>(rand()%11) / double_literal(10.0)));
  }
  weight_updater.update_solution_with_weights();

  check_if_the_same(*net, *solution);

  /* Change a single weight and take it over into the generated solution */
  for(uint32 variant = 0; variant < 10; ++variant){
    uint32 weight_index = rand()%(net->weight_table_size());
    net->set_weight_table(weight_index,(static_cast<sdouble32>(rand()%11) / double_literal(10.0)));
    weight_updater.update_solution_with_weight(weight_index);
    check_if_the_same(*net, *solution);
  }

}

} /* namespace rafko_net_test */
