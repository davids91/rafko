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
#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"

#include "services/sparse_net_builder.h"
#include "services/synapse_iterator.h"
#include "services/backpropagation_queue_wrapper.h"
#include "services/neuron_router.h"

#include <memory>

namespace sparse_net_library_test {
  using std::unique_ptr;
  using std::make_unique;
  using std::vector;

  using sparse_net_library::uint32;
  using sparse_net_library::SparseNet;
  using sparse_net_library::Sparse_net_builder;
  using sparse_net_library::Synapse_iterator;
  using sparse_net_library::Synapse_interval;
  using sparse_net_library::COST_FUNCTION_MSE;
  using sparse_net_library::Backpropagation_queue;
  using sparse_net_library::Backpropagation_queue_wrapper;
  using sparse_net_library::Neuron_router;

/*###############################################################################################
 * Testing Backpropagation order:
 * - Backpropagation queue shall set an order of calculation for the Neurons
 * - for each neuron in the backpropagation queue:
 *   no inputs(dependencies) can have a lower order of the back-propagation
 *   that means no input of a neuron shall be calulated before it
 * */
TEST_CASE( "Testing backpropagation queue", "" ) {
  unique_ptr<Sparse_net_builder> builder(make_unique<Sparse_net_builder>());
  builder->input_size(10).expected_input_range(5.0L).cost_function(COST_FUNCTION_MSE);

  SparseNet* net(builder->dense_layers({20,10,3,5,5}));
  Neuron_router router(*net);

  /* Create a backrpop queue */
  Backpropagation_queue_wrapper queue_wrapper(*net);
  Backpropagation_queue queue = queue_wrapper();

  /* Check integrity */
  vector<uint32> neuron_depth = vector<uint32>(net->neuron_array_size(), 0);
  uint32 num_neurons = 0;
  uint32 current_depth = 0;
  uint32 current_row = 0;
  REQUIRE( 0 < Synapse_iterator(queue.neuron_synapses()).size() );
  Synapse_iterator(queue.neuron_synapses()).iterate([&](int neuron_index){
    REQUIRE( net->neuron_array_size() > neuron_index ); /* all indexes shall be inside network bounds */
    ++num_neurons;
    neuron_depth[neuron_index] = current_depth;
    ++current_row;

    REQUIRE( queue.cols_size() > current_depth ); /* Neuron depth can not be more, than the stored number of depths */
    if(queue.cols(current_depth) <= current_row){
      current_row = 0; /* In case the iteration went through every Neuron in the current depth */
      ++current_depth; /* Increase depth counter */
    }
  });
  CHECK( net->neuron_array_size() == num_neurons ); /* Every Neuron should be found in the Backpropagation queue */

  num_neurons = 0;
  for(int num_cols = 0; num_cols < queue.cols_size(); ++num_cols){
    num_neurons += queue.cols(num_cols);
  }
  CHECK( net->neuron_array_size() == num_neurons ); /* Neuron column numbers shall add up the number of Neurons */

  Synapse_iterator(queue.neuron_synapses()).iterate([&](int neuron_index){
    Synapse_iterator::iterate(net->neuron_array(neuron_index).input_indices(),[=](int input_index){
      if(!Synapse_iterator::is_index_input(input_index))
      CHECK( neuron_depth[neuron_index] < neuron_depth[input_index] );
    });
  });
}

} /* namespace sparse_net_library_test */
