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

#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_net/services/solution_solver.hpp"

#include "test/test_utility.hpp"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Solution generation using the @RafkoNetBuilder and the @SolutionBuilder
 * */
std::unique_ptr<rafko_net::Solution> test_solution_builder_manually(
  google::protobuf::Arena* arena, double device_max_megabytes,
  std::vector<std::uint32_t> net_structure, bool recursion, bool boltzman_knot
){
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
  .set_max_solve_threads(4).set_device_max_megabytes(device_max_megabytes)
  .set_arena_ptr(arena);

  rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings);

  builder.input_size(50).expected_input_range(5.0).output_neuron_number(net_structure.back());

  std::uint32_t layer_index = rand()%net_structure.size();
  if(recursion) builder.add_neuron_recurrence(layer_index, rand()%net_structure[layer_index], 1u);
  if(boltzman_knot) builder.add_feature_to_layer(layer_index, rafko_net::neuron_group_feature_boltzmann_knot);

  rafko_net::RafkoNet* net = nullptr;
  REQUIRE_NOTHROW( net = builder.dense_layers(net_structure) );

  std::unique_ptr<rafko_net::Solution> solution;
  REQUIRE_NOTHROW( solution = rafko_net::SolutionBuilder(settings).build(*net) );

  REQUIRE(net != nullptr);

  CHECK( net->input_data_size() ==   solution->network_input_size() );

  std::int32_t expected_partial_number = 0;
  for(std::int32_t i = 0; i < solution->cols_size(); ++i){
    CHECK( 0 < solution->cols(i) );
    expected_partial_number += solution->cols(i);
  }
  CHECK( expected_partial_number == solution->partial_solutions_size() );

  /* See if every Neuron is inside the result solution */
  bool found;
  for(std::int32_t neuron_iterator = 0; neuron_iterator < net->neuron_array_size(); ++neuron_iterator){
    found = false;
    for(std::int32_t partial_iterator = 0; partial_iterator < solution->partial_solutions_size(); ++partial_iterator){
      REQUIRE( 0u < solution->partial_solutions(partial_iterator).output_data().interval_size() );
      for(
        std::uint32_t internal_neuron_iterator = 0;
        internal_neuron_iterator < solution->partial_solutions(partial_iterator).output_data().interval_size();
        ++internal_neuron_iterator
      ){
        if(static_cast<std::int32_t>(solution->partial_solutions(partial_iterator).output_data().starts() + internal_neuron_iterator) == neuron_iterator){
          found = true;
          goto Solution_search_over; /* don't judge me */
        }
      }
    }
    Solution_search_over:
    REQUIRE( true == found ); /* Found the Neuron index from the net in the result solution */
  }

  /* Test if the inputs of the partial in the first row only contain input indexes */
  rafko_test::check_if_the_same(*net, *solution);

  if(nullptr == arena){
    delete net;
  }

  return solution;
}

TEST_CASE( "Building a solution from a small net", "[build][small][build-only]" ){
  double space_used_megabytes = 0;
  std::unique_ptr<rafko_net::Solution> solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, 2048.0, {2,2,3,1,2}, false, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */;
  solution.reset();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, space_used_megabytes/5.0, {2,2,3,1,2}, false, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  solution.reset();

  /* again, but with recursion enabled */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, 2048.0, {20,20,30,10,5}, true, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */;
  solution.reset();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, space_used_megabytes/5.0,{2,2,3,1,2}, true, false
  ));

  /* again, but with boltzmann recursion enabled */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, 2048.0, {20,20,30,10,5}, true, true
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */;
  solution.reset();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, space_used_megabytes/5.0,{2,2,3,1,2}, true, true
  ));

  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
}

TEST_CASE( "Building a solution from a bigger net", "[build][build-only]" ){
  double space_used_megabytes = 0;
  std::unique_ptr<rafko_net::Solution> solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, 2048.0, {20,20,30,10,5}, false, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */;
  solution.reset();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr,space_used_megabytes/5.0,{20,20,30,10,5}, false, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  solution.reset();

  /* again, but With recursion enabled */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, 2048.0, {20,20,30,10,5}, true, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// (1024.0) /* KB *// (1024.0) /* MB */;
  solution.reset();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = std::unique_ptr<rafko_net::Solution>(test_solution_builder_manually(
    nullptr, space_used_megabytes/5.0, {20,20,30,10,5}, true, false
  ));
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
}

} /* namespace rafko_net_test */
