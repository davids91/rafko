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
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_gym/services/rafko_weight_updater.h"

#include "test/test_utility.h"

namespace rafko_gym_test {


TEST_CASE("Testing if weight updater updates weights of a solution sufficiently even in bulk","[weight_updater][weight-update][bulk]"){
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_learning_rate(0.1);
  std::vector<std::uint32_t> net_structure = {2,4,3,1,2};
  std::vector<double> net_input = {(10.0),(20.0),(30.0),(40.0),(50.0)};
  std::unique_ptr<rafko_net::RafkoNet> network(rafko_net::RafkoNetBuilder(settings).input_size(5).expected_input_range((5.0)).dense_layers(net_structure));
  rafko_gym::RafkoWeightUpdater weight_updater(*network, settings);

  /* Change the weights in the network and take them over into the generated solution */
  for(std::uint32_t variant = 0; variant < 10; ++variant){

    std::vector<double> weight_deltas(network->weight_table_size());
    std::generate(weight_deltas.begin(), weight_deltas.end(), [](){
      return static_cast<double>(rand()%100) / (100.0);
    });

    /* calculate weight supposed values */
    std::uint32_t i = 0u;
    std::vector<double> weight_references(network->weight_table_size());
    std::generate(weight_references.begin(), weight_references.end(), [&i, &network, weight_deltas, settings](){
      ++i;
      double ref_w = network->weight_table(i-1) - (weight_deltas[i-1] * settings.get_learning_rate());
      return ref_w;
    });

    if(weight_updater.is_finished())weight_updater.start();
    weight_updater.iterate(weight_deltas);

    for(std::int32_t i = 0; i < network->weight_table_size(); ++i){
      REQUIRE(  Catch::Approx(weight_references[i]).epsilon(0.00000000000001) == network->weight_table(i) );
    }
  }
}

} /* namespace rafko_gym_test */
