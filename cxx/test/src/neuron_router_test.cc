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

#include "gen/sparse_net.pb.h"
#include "services/sparse_net_builder.h"

#include "gen/solution.pb.h"
#include "services/solution_builder.h"

#include "services/neuron_router.h"

namespace sparse_net_library_test {

using std::unique_ptr;
using std::make_unique;
using std::deque;
using std::vector;

using sparse_net_library::uint16;
using sparse_net_library::uint32;
using sparse_net_library::Sparse_net_builder;
using sparse_net_library::SparseNet;
using sparse_net_library::Neuron_router;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::Synapse_iterator;

/*###############################################################################################
 * Testing if the iteration is correctly processing the Sparse net
 * - Building a Fully Connected Multilayered Net
 * - Each iteration has to add the corresponding layer
 *    Because of the structure of a fully connected Net, one iteration would involve one layer exactly
 * */
TEST_CASE( "Testing Neural Network Iteration Routing", "[neuron-iteration][small]" ){
  /* Build a net */
  vector<uint32> layer_structure = {2,3,3,5};
  unique_ptr<Sparse_net_builder> net_builder = make_unique<Sparse_net_builder>();
  net_builder->input_size(5).output_neuron_number(5)
  .cost_function(COST_FUNCTION_MSE).expected_input_range(double_literal(5.0));
  SparseNet* net(net_builder->dense_layers(layer_structure));
  net_builder.reset();
  Neuron_router net_iterator(*net);

  /* Testing the collected subset in each iteration in the net */
  uint16 iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */

  uint32 layer_start = 0;
  uint32 tmp_index;
  bool last_run = false;
  CHECK( false == net_iterator.finished() );
  while(!net_iterator.finished()){ /* Until the whole output layer is processed */
    net_iterator.collect_subset(iteration,1,double_literal(500.0));

    /* For a fully connected Dense Layer, each iteration subset should be the actual layer */
    vector<uint32> subset;
    while(net_iterator.get_first_neuron_index_from_subset(tmp_index)){
      subset.push_back(tmp_index);
      net_iterator.confirm_first_subset_element_processed(tmp_index);
    }
    REQUIRE((
      (iteration <= layer_structure.size()) /* Has to finish sooner, than there are layers */
      ||((0 == subset.size())&&(!last_run)) /* With the exception of the last iteration */
    )); /* ..where only the output_layer_iterator  is updated to the end */
    /*!Note: Iteration starts from 1! so equality is needed here */
    if(0 < subset.size()){
      for(uint32 i = 0; i < layer_structure[iteration-1]; ++i){ /* Find all indexes inside the layer in the current subset */
        CHECK( std::find(subset.begin(), subset.end(), layer_start + i) != subset.end() );
      }
    }else{
      last_run = true;
    }
    if( layer_structure.size() > iteration ) /* iteration needs to run an additional round, */
      layer_start += layer_structure[iteration-1]; /* so this way OOB can be avoided */

    subset.clear();

     ++iteration;
  }

}

} /* namespace sparse_net_library_test */
