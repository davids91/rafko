#include "test/catch.hpp"

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "models/neuron_info.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library_test {

  using sparse_net_library::Neuron;
  using sparse_net_library::Neuron_info;
  using sparse_net_library::Synapse_iterator;
  using sparse_net_library::Synapse_interval;
  using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;

/*###############################################################################################
 * Testing Neuron Validation
 * */
TEST_CASE( "Testing Neuron validation", "[Neuron][manual]" ) {

  Synapse_interval temp_synapse_interval;

  /* Empty Neuron should be invalid */
  Neuron neuron = Neuron();
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  /* Setting some parameters */
  neuron.set_bias_idx(0); /* Unfortunately checking against the weight table is not possible without Net context */
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  neuron.set_memory_filter_idx(0);
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  neuron.set_transfer_function_idx(TRANSFER_FUNCTION_IDENTITY);
  CHECK( true == Neuron_info::is_neuron_valid(neuron) );

  /* Setting indexing information */
  temp_synapse_interval.set_starts(0); /* Adding weight inputs */
  temp_synapse_interval.set_interval_size(0);
  *neuron.add_input_weights() = temp_synapse_interval;
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  temp_synapse_interval.set_starts(0); /* Adding an index input of a non-matching number */
  temp_synapse_interval.set_interval_size(5);
  *neuron.add_input_indices() = temp_synapse_interval;
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  temp_synapse_interval.set_starts(0); /* Extending input indices to match weights */
  temp_synapse_interval.set_interval_size(4);
  *neuron.add_input_weights() = temp_synapse_interval;
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );
  
  temp_synapse_interval.set_starts(4);
  temp_synapse_interval.set_interval_size(1);
  *neuron.add_input_weights() = temp_synapse_interval;
  CHECK( true == Neuron_info::is_neuron_valid(neuron) );

  temp_synapse_interval.set_starts(5); /* Adding additional weights */
  temp_synapse_interval.set_interval_size(5);
  *neuron.add_input_weights() = temp_synapse_interval;
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  temp_synapse_interval.set_starts(5); /* Indices to follow number of weights */
  temp_synapse_interval.set_interval_size(3);
  *neuron.add_input_indices() = temp_synapse_interval;
  CHECK( false == Neuron_info::is_neuron_valid(neuron) );

  temp_synapse_interval.set_starts(8); /* Indices to follow number of weights */
  temp_synapse_interval.set_interval_size(2);
  *neuron.add_input_indices() = temp_synapse_interval;
  CHECK( true == Neuron_info::is_neuron_valid(neuron) );
}

} /* namespace sparse_net_library_test */