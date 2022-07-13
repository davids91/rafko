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
#include <vector>
#include <memory>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/solution.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_gym/services/rafko_weight_adapter.hpp"

#include "test/test_utility.hpp"

namespace rafko_gym_test {

TEST_CASE("Testing if weight adapter updates weights of a solution sufficiently","[weight-adapter][weight-update]"){
  rafko_mainframe::RafkoSettings settings;
  std::vector<std::uint32_t> net_structure = {2,4,3,1,2};
  std::vector<double> net_input = {(10.0),(20.0),(30.0),(40.0),(50.0)};
  std::unique_ptr<rafko_net::RafkoNet> net(rafko_net::RafkoNetBuilder(settings).input_size(5).expected_input_range((5.0)).dense_layers(net_structure));
  std::unique_ptr<rafko_net::Solution> solution = std::unique_ptr<rafko_net::Solution>(rafko_net::SolutionBuilder(settings).build(*net));
  rafko_gym::RafkoWeightAdapter weight_adapter(*net, *solution, settings);
  rafko_test::check_if_the_same(*net, *solution);

  /* Change the weights in the network and take them over into the generated solution */
  srand (time(nullptr));
  for(std::int32_t weight_iterator = 0; weight_iterator < net->weight_table_size() ; ++weight_iterator){
    net->set_weight_table(weight_iterator,(static_cast<double>(rand()%11) / (10.0)));
  }
  weight_adapter.update_solution_with_weights();

  rafko_test::check_if_the_same(*net, *solution);

  /* Change a single weight and take it over into the generated solution */
  for(std::uint32_t variant = 0; variant < 10; ++variant){
    std::uint32_t weight_index = rand()%(net->weight_table_size());
    net->set_weight_table(weight_index,(static_cast<double>(rand()%11) / (10.0)));
    weight_adapter.update_solution_with_weight(weight_index);
    rafko_test::check_if_the_same(*net, *solution);
  }
}


} /* namespace rafko_gym_test */
