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
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_net/services/neuron_router.hpp"

#include "test/test_utility.hpp"

namespace rafko_net_test {

/*###############################################################################################
 * Testing if the iteration is correctly processing the Sparse net
 * - Building a Fully Connected Multilayered Net
 * - Each iteration has to add the corresponding layer
 *    Because of the structure of a fully connected Net, one iteration would involve one layer exactly
 * */
TEST_CASE( "Testing Neural Network Iteration Routing", "[neuron-iteration][small]" ){
  rafko_mainframe::RafkoSettings settings;
  /* Build a net and router */
  std::vector<std::uint32_t> layer_structure = {2,3,3,5};
  std::unique_ptr<rafko_net::RafkoNetBuilder> net_builder = std::make_unique<rafko_net::RafkoNetBuilder>(settings);
  net_builder->input_size(5).output_neuron_number(5).expected_input_range((5.0));
  std::unique_ptr<rafko_net::RafkoNet> net = std::unique_ptr<rafko_net::RafkoNet>(net_builder->dense_layers(layer_structure));
  net_builder.reset();
  rafko_net::NeuronRouter net_iterator(*net);

  /* Testing the collected subset in each iteration in the net */
  std::uint16_t iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */

  std::uint32_t layer_start = 0;
  std::uint32_t neuron_index;
  std::vector<std::uint32_t>::iterator neuron_in_subset;
  bool found = false;
  bool last_run = false;
  CHECK( false == net_iterator.finished() );
  while(!net_iterator.finished()){ /* Until the whole output layer is processed */
    net_iterator.collect_subset(1,(500.0),true);

    /* For a fully connected Dense Layer, each iteration subset should be the actual layer */
    std::vector<std::uint32_t> subset;
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
      for(std::uint32_t i = 0; i < layer_structure[iteration-1]; ++i){ /* Find all indexes inside the layer in the current subset */
        neuron_in_subset = std::find(subset.begin(), subset.end(), layer_start + i);
        REQUIRE( neuron_in_subset != subset.end() );

        /* And check its dependencies */
        rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::iterate(net->neuron_array(layer_start + i).input_indices(),
        [&](std::int32_t synapse_input_index){
          if(
            (!rafko_net::SynapseIterator<>::is_index_input(synapse_input_index)) /* Every net-internal Neuron input.. */
            &&(!net_iterator.is_neuron_processed(synapse_input_index)) /* ..should be already solved.. */
          ){
            found = false;
            for(std::vector<std::uint32_t>::iterator iter = subset.begin(); iter != neuron_in_subset; ++iter){
              if(static_cast<std::int32_t>(*iter) == synapse_input_index){
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
  rafko_mainframe::RafkoSettings settings;

  /* Build a net and router */
  std::vector<std::uint32_t> layer_structure = {2,3,3,5};
  std::unique_ptr<rafko_net::RafkoNet> net(
    rafko_net::RafkoNetBuilder(settings)
    .input_size(5).output_neuron_number(5)
    .expected_input_range((5.0))
    .dense_layers(layer_structure)
  );
  rafko_net::NeuronRouter net_iterator(*net);

  /* Collect the whole network into one big subset */
  while(static_cast<std::int32_t>(net_iterator.get_subset_size()) < net->neuron_array_size()){ /* Until the whole network is processed */
    net_iterator.collect_subset(1,(500.0),false);
  }

  /* Go through the second layer of the network */
  /* All Neurons in the current layer should display to be without any dependency */
  for(std::uint32_t i = 0; i < layer_structure[1]; ++i)
    REQUIRE( true == net_iterator.is_neuron_without_dependency(layer_structure[0] + i) );

  /* Omit some Neurons from the previous layer */
  for(std::uint32_t i = 0; i < layer_structure[0]; ++i){
    if(0 == (i%2)){
      net_iterator.confirm_first_subset_element_ommitted(i);
    }
  }

  /* No Neurons in the current layer should display to be without any dependency now */
  for(std::uint32_t i = 0; i < layer_structure[1]; ++i){
    REQUIRE( false == net_iterator.is_neuron_without_dependency(layer_structure[0] + i) );
  }
}

} /* namespace rafko_net_test */
