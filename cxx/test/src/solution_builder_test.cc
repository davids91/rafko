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

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/solution_solver.h"

namespace sparse_net_library_test {

using sparse_net_library::Sparse_net_builder;
using sparse_net_library::SparseNet;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution;
using sparse_net_library::Solution_solver;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::NETWORK_RECURRENCE_TO_SELF;
using sparse_net_library::NETWORK_RECURRENCE_TO_LAYER;
using rafko_mainframe::Service_context;

using std::unique_ptr;
using std::shared_ptr;
using std::make_unique;
using std::vector;

/*###############################################################################################
 * Testing Solution generation using the @Sparse_net_builder and the @Solution_builder
 * */
unique_ptr<Solution> test_solution_builder_manually(google::protobuf::Arena* arena, sdouble32 device_max_megabytes, uint32 recursion){
  Service_context service_context;

  vector<uint32> net_structure = {20,20,30,10,2}; /* Build a net of this structure */
  Sparse_net_builder builder = Sparse_net_builder(service_context);

  builder.input_size(50).expected_input_range(double_literal(5.0))
    .output_neuron_number(2).arena_ptr(arena);

  if(NETWORK_RECURRENCE_TO_SELF == recursion){
    builder.set_recurrence_to_self();
  }else if(NETWORK_RECURRENCE_TO_LAYER == recursion){
    builder.set_recurrence_to_layer();
  }

  SparseNet net;
  REQUIRE_NOTHROW(
    net = *builder.dense_layers(net_structure)
  );

   unique_ptr<Solution> solution;

   REQUIRE_NOTHROW(
      solution = unique_ptr<Solution>(Solution_builder(service_context)
        .max_solve_threads(4).device_max_megabytes(device_max_megabytes)
        .arena_ptr(arena).build(net)
      )
   );

  /* See if every Neuron is inside the result solution */
  bool found;
  for(sint32 neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){
    found = false;
    for(sint32 partial_solution_iterator = 0; partial_solution_iterator < solution->partial_solutions_size(); ++partial_solution_iterator){
      Synapse_iterator<> output_neurons(solution->partial_solutions(partial_solution_iterator).output_data());
      for(
        uint32 internal_neuron_iterator = 0;
        internal_neuron_iterator < solution->partial_solutions(partial_solution_iterator).internal_neuron_number();
        ++internal_neuron_iterator
      ){
        if(output_neurons[internal_neuron_iterator] == neuron_iterator){
          found = true;
          goto Solution_search_over; /* don't judge */
        }
      }
    }
    Solution_search_over:
    CHECK( true == found ); /* Found the Neuron index from the net in the result solution */
  }

  /* Test if the inputs of the partial in the first row only contain input indexes */
  check_if_the_same(net, *solution);

  /* TODO: Test if all of the neuron is present in all of the partial solutions outputs */
  return solution;
}

TEST_CASE( "Building a solution from a net", "[build][small][build-only]" ){
  sdouble32 space_used_megabytes = 0;
  unique_ptr<Solution> solution = test_solution_builder_manually(nullptr,double_literal(2048.0),0);
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */;
  solution.release();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = test_solution_builder_manually(nullptr,space_used_megabytes/double_literal(5.0),0);
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  solution.release();

  /* again, but With recursion enabled */
  solution = test_solution_builder_manually(nullptr,double_literal(2048.0),0x02);
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */;
  solution.release();

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = test_solution_builder_manually(nullptr,space_used_megabytes/double_literal(5.0),0x02);
  REQUIRE( nullptr != solution );
  REQUIRE( 0 < solution->SpaceUsedLong() );
  solution.release();
}

} /* namespace sparse_net_library_test */
