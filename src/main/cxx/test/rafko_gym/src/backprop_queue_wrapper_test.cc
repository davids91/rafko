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

#include <memory>
#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/neuron_router.h"

#include "rafko_gym/services/backpropagation_queue_wrapper.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

/*###############################################################################################
 * Testing Backpropagation order:
 * - Backpropagation queue shall set an order of calculation for the Neurons
 * - for each neuron in the backpropagation queue:
 *   no inputs(dependencies) can have a lower order of the back-propagation
 *   that means no input of a neuron shall be calulated before it
 * */
TEST_CASE( "Testing backpropagation queue", "[build][backprop]" ) {
  rafko_mainframe::RafkoServiceContext service_context;
  std::unique_ptr<rafko_net::RafkoNetBuilder> builder(std::make_unique<rafko_net::RafkoNetBuilder>(service_context));
  builder->input_size(10).expected_input_range(double_literal(5.0));

  std::unique_ptr<rafko_net::RafkoNet> net(builder->dense_layers({20,10,3,5,5}));
  rafko_net::NeuronRouter router(*net);

  /* Create a backrpop queue */
  rafko_gym::BackpropagationQueueWrapper queue_wrapper(*net, service_context);
  rafko_gym::BackpropagationQueue queue = queue_wrapper();

  /* Check integrity */
  std::vector<uint32> neuron_depth = std::vector<uint32>(net->neuron_array_size(), 0);
  uint32 num_neurons = 0;
  uint32 current_depth = 0;
  uint32 current_row = 0;
  REQUIRE( 0 < rafko_net::SynapseIterator<>(queue.neuron_synapses()).size() );
  rafko_net::SynapseIterator<>::iterate(queue.neuron_synapses(),[&](sint32 neuron_index){
    REQUIRE( net->neuron_array_size() > neuron_index ); /* all indexes shall be inside network bounds */
    ++num_neurons;
    neuron_depth[neuron_index] = current_depth;
    ++current_row;

    REQUIRE( queue.cols_size() > static_cast<sint32>(current_depth) ); /* Neuron depth can not be more, than the stored number of depths */
    if(queue.cols(current_depth) <= current_row){
      current_row = 0; /* In case the iteration went through every Neuron in the current depth */
      ++current_depth; /* Increase depth counter */
    }
  });
  CHECK( net->neuron_array_size() == static_cast<sint32>(num_neurons) ); /* Every Neuron should be found in the Backpropagation queue */

  num_neurons = 0;
  for(int num_cols = 0; num_cols < queue.cols_size(); ++num_cols){
    num_neurons += queue.cols(num_cols);
  }
  CHECK( net->neuron_array_size() == static_cast<sint32>(num_neurons) ); /* Neuron column numbers shall add up the number of Neurons */

  rafko_net::SynapseIterator<>::iterate(queue.neuron_synapses(),[&](sint32 neuron_index){
    rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>::iterate(net->neuron_array(neuron_index).input_indices(),
    [=](sint32 input_index){
      if(!rafko_net::SynapseIterator<>::is_index_input(input_index))
      CHECK( neuron_depth[neuron_index] < neuron_depth[input_index] );
    });
  });
}

} /* namespace rafko_gym_test */
