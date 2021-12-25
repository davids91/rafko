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
#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/synapse_iterator.h"

#include "test/test_utility.h"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Neuron Validation
 * */
TEST_CASE( "Testing Neuron validation", "[Neuron][manual]" ) {

  rafko_net::IndexSynapseInterval temp_index_interval;
  rafko_net::InputSynapseInterval temp_input_interval;

  /* Empty Neuron should be invalid */
  rafko_net::Neuron neuron = rafko_net::Neuron();
  CHECK( false == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  /* Setting some parameters */
  /* Unfortunately checking against the weight table is not possible without Net context */
  CHECK( false == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  neuron.set_transfer_function_idx(rafko_net::transfer_function_identity);
  CHECK( true == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  /* Setting indexing information */
  temp_index_interval.set_starts(0); /* Adding weight inputs */
  temp_index_interval.set_interval_size(0);
  *neuron.add_input_weights() = temp_index_interval;
  CHECK( false == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_input_interval.set_starts(0); /* Adding an index input of a non-matching number */
  temp_input_interval.set_interval_size(5);
  *neuron.add_input_indices() = temp_input_interval;
  CHECK( false == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_index_interval.set_starts(0); /* Extending input indices to match weights */
  temp_index_interval.set_interval_size(4);
  *neuron.add_input_weights() = temp_index_interval;
  CHECK( false == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_index_interval.set_starts(4);
  temp_index_interval.set_interval_size(1);
  *neuron.add_input_weights() = temp_index_interval;
  CHECK( true == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_index_interval.set_starts(5); /* Adding additional weights */
  temp_index_interval.set_interval_size(5); /* ..should still be a valid state, since the extra weights count as biases */
  *neuron.add_input_weights() = temp_index_interval;
  CHECK( true == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_input_interval.set_starts(5); /* Indices to follow number of weights */
  temp_input_interval.set_interval_size(3);
  *neuron.add_input_indices() = temp_input_interval;
  CHECK( true == rafko_net::NeuronInfo::is_neuron_valid(neuron) );

  temp_input_interval.set_starts(8); /* Indices to follow number of weights */
  temp_input_interval.set_interval_size(2);
  *neuron.add_input_indices() = temp_input_interval;
  CHECK( true == rafko_net::NeuronInfo::is_neuron_valid(neuron) );
}

} /* namespace rafko_net_test */
