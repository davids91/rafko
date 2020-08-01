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

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/neuron_router.h"

namespace sparse_net_library_test {

using std::unique_ptr;
using std::make_unique;
using std::deque;
using std::vector;

using sparse_net_library::Sparse_net_builder;
using sparse_net_library::SparseNet;
using sparse_net_library::Neuron_router;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Input_synapse_interval;
using rafko_mainframe::Service_context;

/*###############################################################################################
 * Testing if the iteration is correctly processing the Sparse net
 * - Building a Fully Connected Multilayered Net
 * - Each iteration has to add the corresponding layer
 *    Because of the structure of a fully connected Net, one iteration would involve one layer exactly
 * */
TEST_CASE( "Testing Neural Network Iteration Routing", "[neuron-iteration][small]" ){
  Service_context service_context;
  /* Build a net and router */
  vector<uint32> layer_structure = {2,3,3,5};
  unique_ptr<Sparse_net_builder> net_builder = make_unique<Sparse_net_builder>(service_context);
  net_builder->input_size(5).output_neuron_number(5).expected_input_range(double_literal(5.0));
  SparseNet* net(net_builder->dense_layers(layer_structure));
  net_builder.reset();
  Neuron_router net_iterator(*net);

  /* Testing the collected subset in each iteration in the net */
  uint16 iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */

  uint32 layer_start = 0;
  uint32 neuron_index;
  vector<uint32>::iterator neuron_in_subset;
  bool found = false;
  bool last_run = false;
  CHECK( false == net_iterator.finished() );
  while(!net_iterator.finished()){ /* Until the whole output layer is processed */
    net_iterator.collect_subset(1,double_literal(500.0),true);

    /* For a fully connected Dense Layer, each iteration subset should be the actual layer */
    vector<uint32> subset;
    while(net_iterator.get_first_neuron_index_from_subset(neuron_index)){
      subset.push_back(neuron_index);
      net_iterator.confirm_first_subset_element_processed(neuron_index);
    }
    REQUIRE((
      (iteration <= layer_structure.size()) /* Has to finish sooner, than there are layers */
      ||((0 == subset.size())&&(!last_run)) /* With the exception of the last iteration */
    )); /* ..where only the output_layer_iterator  is updated to the end */
    /*!Note: Iteration starts from 1! so equality is needed here */
    if(0 < subset.size()){
      for(uint32 i = 0; i < layer_structure[iteration-1]; ++i){ /* Find all indexes inside the layer in the current subset */
        neuron_in_subset = std::find(subset.begin(), subset.end(), layer_start + i);
        REQUIRE( neuron_in_subset != subset.end() );

        /* And check its dependencies */
        Synapse_iterator<Input_synapse_interval>::iterate(net->neuron_array(layer_start + i).input_indices(),
        [&](Input_synapse_interval input_synapse, sint32 synapse_input_index){
          if( /* Every net-internal Neuron input.. */
            (!Synapse_iterator<>::is_index_input(synapse_input_index))
            &&(!net_iterator.is_neuron_processed(synapse_input_index))
          ){ /* ..should be already solved.. */
            found = false;
            for(vector<uint32>::iterator iter = subset.begin(); iter != neuron_in_subset; ++iter){
              if(static_cast<sint32>(*iter) == synapse_input_index){
                found = true;
                break;
              }
            }
            REQUIRE( true == found ); /* .. or must be found before its parent in the subset, if it's not processed already */
          }
        });
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

/*###############################################################################################
 * Testing if the dependency calculations are correct inside the interface @is_neuron_without_dependency
 *  by building a Neuron network, ommiting neurons from the subset, and then checking return values
 * */
TEST_CASE( "Testing Neural Network router dependency interface", "[neuron-iteration][neuron-dependency]" ){
  Service_context service_context;

  /* Build a net and router */
  vector<uint32> layer_structure = {2,3,3,5};
  unique_ptr<SparseNet> net(
    Sparse_net_builder(service_context)
    .input_size(5).output_neuron_number(5)
    .expected_input_range(double_literal(5.0))
    .dense_layers(layer_structure)
  );
  Neuron_router net_iterator(*net);

  /* Collect the whole network into one big subset */
  while(static_cast<sint32>(net_iterator.get_subset_size()) < net->neuron_array_size()){ /* Until the whole network is processed */
    net_iterator.collect_subset(1,double_literal(500.0),false);
  }

  /* Go through the second layer of the network */
  /* All Neurons in the current layer should display to be without any dependency */
  for(uint32 i = 0; i < layer_structure[1]; ++i)
    REQUIRE( true == net_iterator.is_neuron_without_dependency(layer_structure[0] + i) );

  /* Omit some Neurons from the previous layer */
  for(uint32 i = 0; i < layer_structure[0]; ++i){
    if(0 == (i%2)){
      net_iterator.confirm_first_subset_element_ommitted(i);
    }
  }

  /* No Neurons in the current layer should display to be without any dependency now */
  for(uint32 i = 0; i < layer_structure[1]; ++i){
    REQUIRE( false == net_iterator.is_neuron_without_dependency(layer_structure[0] + i) );
  }
}

} /* namespace sparse_net_library_test */
