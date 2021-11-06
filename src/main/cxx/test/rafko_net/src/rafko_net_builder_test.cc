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

#include <memory>

#include "rafko_protocol/common.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net_test {

using rafko_net::Neuron;
using rafko_net::RafkoNet;
using rafko_net::Transfer_functions;
using rafko_net::RafkoNet_builder;
using rafko_net::Neuron_info;
using rafko_net::transfer_function_identity;
using rafko_net::transfer_function_sigmoid;
using rafko_net::transfer_function_tanh;
using rafko_net::transfer_function_relu;
using rafko_net::transfer_function_selu;
using rafko_net::InputSynapseInterval;
using rafko_net::IndexSynapseInterval;
using rafko_net::Synapse_iterator;
using rafko_mainframe::Service_context;

using std::make_shared;
using std::shared_ptr;
using std::unique_ptr;
using std::make_unique;
using std::vector;


/*###############################################################################################
 * Testing Manual Net creation
 * Create 3 Neurons, each having the same weight
 * The 0th Neuron shall have the input, which is a number
 * the last 2 shall be the outputs
 * 0th Neuron shall have 5 inputs
 * 1st and 2nd neurons will have the first as input both
 * */
RafkoNet* test_net_builder_manually(google::protobuf::Arena* arena){
  Service_context service_context = Service_context().set_arena_ptr(arena);

  /* Create the single Weight Table */
  IndexSynapseInterval temp_index_interval;
  InputSynapseInterval temp_input_interval;
  sdouble32 used_weight = double_literal(0.5);
  Transfer_functions used_transfer_function = transfer_function_sigmoid;
  vector<sdouble32> weight_table {double_literal(0.0),double_literal(0.0)};
  weight_table[0] = used_weight;
  REQUIRE( nullptr != &(weight_table[0]) );
  REQUIRE( weight_table[0] == used_weight);

  /* Create the Neuron Table */
  vector<Neuron> neuron_table(3);

  /* Neuron 0 Has an input of 1 */
  neuron_table[0].set_transfer_function_idx(used_transfer_function);
  neuron_table[0].set_memory_filter_idx(0); /* Weight 0 in the weight_table */
  temp_input_interval.set_starts(0); /* Input Starting from 0 */
  temp_input_interval.set_interval_size(1); /* 1 Input */
  *neuron_table[0].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0); /* Weight 0 in the weight_table */
  temp_index_interval.set_interval_size(2); /* Weight0 + bias0 in the weight_table */
  *neuron_table[0].add_input_weights() = temp_index_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid( neuron_table[0]) );

  /* Neuron 1 Has Neuron 0 as input */
  neuron_table[1].set_transfer_function_idx(used_transfer_function);
  neuron_table[1].set_memory_filter_idx(0);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(1);
  *neuron_table[1].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(2);
  *neuron_table[1].add_input_weights() = temp_index_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid( neuron_table[1]) );

  /* Neuron 2 Also has Neuron 0 as input */
  neuron_table[2].set_transfer_function_idx(used_transfer_function);
  neuron_table[2].set_memory_filter_idx(0);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(1);
  *neuron_table[2].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(2);
  *neuron_table[2].add_input_weights() = temp_index_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid( neuron_table[2]) );

  /* Pass the net into the builder */
  shared_ptr<RafkoNet_builder> builder(make_shared<RafkoNet_builder>(service_context));
  builder->input_size(1).expected_input_range(double_literal(1.0)).output_neuron_number(2)
    .neuron_array(neuron_table).weight_table(weight_table);

  try{
    /* Build the net with the given parameters */
    RafkoNet* net(builder->build());

    /* Check Net parameters */
    REQUIRE( 0 < net->neuron_array_size() );
    REQUIRE( 0 < net->weight_table_size() );
    CHECK( 3 == net->neuron_array_size() );
    CHECK( 2 == net->weight_table_size() );
    CHECK( used_weight == net->weight_table(0) );

    /* Check parameters for each neuron */
    REQUIRE( true == Neuron_info::is_neuron_valid(net->neuron_array(0)) );
    REQUIRE( 0 < net->neuron_array(0).input_indices_size() );
    CHECK( 1 == net->neuron_array(0).input_indices_size() );
    CHECK( 1 == net->neuron_array(0).input_indices(0).interval_size() );
    CHECK( 0 == net->neuron_array(0).input_indices(0).starts() );
    REQUIRE( 0 < net->neuron_array(0).input_weights_size() );
    CHECK( 1 == net->neuron_array(0).input_weights_size() );
    CHECK( 2 == net->neuron_array(0).input_weights(0).interval_size() );
    CHECK( 0 == net->neuron_array(0).input_weights(0).starts() );
    CHECK(
      weight_table[net->neuron_array(0).input_weights(0).starts()]
      == net->weight_table(net->neuron_array(0).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(0).transfer_function_idx()
    );

    REQUIRE( true == Neuron_info::is_neuron_valid(net->neuron_array(1)) );
    REQUIRE( 0 < net->neuron_array(1).input_indices_size() );
    CHECK( 1 == net->neuron_array(1).input_indices_size() );
    CHECK( 1 == net->neuron_array(1).input_indices(0).interval_size() );
    CHECK( 0 == net->neuron_array(1).input_indices(0).starts() );
    REQUIRE( 0 < net->neuron_array(1).input_weights_size() );
    CHECK( 1 == net->neuron_array(1).input_weights_size() );
    CHECK( 2 == net->neuron_array(1).input_weights(0).interval_size() );
    CHECK( 0 == net->neuron_array(1).input_weights(0).starts() );
    CHECK(
      weight_table[net->neuron_array(1).input_weights(0).starts()]
      == net->weight_table(net->neuron_array(1).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(1).transfer_function_idx()
    );

    REQUIRE( true == Neuron_info::is_neuron_valid(net->neuron_array(2)) );
    REQUIRE( 0 < net->neuron_array(2).input_indices_size() );
    CHECK( 1 == net->neuron_array(2).input_indices_size() );
    CHECK( 1 == net->neuron_array(2).input_indices(0).interval_size() );
    CHECK( 0 == net->neuron_array(2).input_indices(0).starts() );
    REQUIRE( 0 < net->neuron_array(2).input_weights_size() );
    CHECK( 2 == net->neuron_array(2).input_weights(0).interval_size() );
    CHECK( 0 == net->neuron_array(2).input_weights(0).starts() );
    CHECK(
      weight_table[net->neuron_array(2).input_weights(0).starts()]
      == net->weight_table(net->neuron_array(2).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(2).transfer_function_idx()
    );
    return net;
  }catch(const char* e){
    return nullptr;
  }
}

TEST_CASE( "Constructing small net manually", "[build][small][manual]" ) {
  RafkoNet* net = test_net_builder_manually(nullptr);
  REQUIRE( nullptr != net );
  delete net;
}

TEST_CASE("Constructing small net manually using arena","[build][arena][small][manual]"){
  google::protobuf::Arena arena;
  RafkoNet* net = test_net_builder_manually(&arena);
  REQUIRE( nullptr != net );
  arena.Reset();
}

/*###############################################################################################
 * Testing Fully Connected Net creation
 * Create a small neural network of 7 Neurons and 5 inputs:
 * -Input Layer: 2 Neurons
 * -Hidden Layer: 3 Neurons
 * -Output Layer: 2 Neurons
 * And check manually the connections
 */
RafkoNet* test_net_builder_fully_connected(google::protobuf::Arena* arena){
  Service_context service_context = Service_context().set_arena_ptr(arena);
  unique_ptr<RafkoNet_builder> builder(make_unique<RafkoNet_builder>(service_context));
  builder->input_size(5)
    .output_neuron_number(2)
    .expected_input_range(double_literal(5.0));

  RafkoNet* net(builder->dense_layers(
    {2,3,2},{
      {transfer_function_identity},
      {transfer_function_selu,transfer_function_relu},
      {transfer_function_tanh,transfer_function_sigmoid}
    }
  ));

  /* Check net validity in general */
  REQUIRE( ((5*2+2*2) + (2*3+3*2) + (2*3+2*2)) == net->weight_table_size() );
  REQUIRE( 0 < net->neuron_array_size() );
  CHECK( 7 == net->neuron_array_size() );
  CHECK( 5 == net->input_data_size() );
  CHECK( 2 == net->output_neuron_number() );

  /* Check Neuron validity in general */
  uint32 number_of_input_indexes;
  uint32 number_of_input_weights;
  for(int i = 0; i < 7; ++i){
    REQUIRE( true == Neuron_info::is_neuron_valid(net->neuron_array(i)) );

    /* Check the indexing */
    REQUIRE( 0 < net->neuron_array(i).input_indices_size() );
    CHECK( 1 == net->neuron_array(i).input_indices_size() ); /* One synapse ==> the previous layer */
    number_of_input_indexes = 0;
    for(sint32 index_syn_iter = 0; index_syn_iter < net->neuron_array(i).input_indices_size(); ++index_syn_iter){
      REQUIRE( /* Every index synapse element has to point inside the neuron array */
        net->neuron_array_size() >
        static_cast<sint32>( net->neuron_array(i).input_indices(index_syn_iter).starts()
          + net->neuron_array(i).input_indices(index_syn_iter).interval_size() )
      );
      number_of_input_indexes += net->neuron_array(i).input_indices(index_syn_iter).interval_size();
    }


    /* Check Weight indexes */
    number_of_input_weights = 0;
    REQUIRE( 0 < net->neuron_array(i).input_weights_size() );
    for(sint32 weight_syn_iterator = 0; weight_syn_iterator < net->neuron_array(i).input_weights_size(); ++weight_syn_iterator){
      /* Bias and memory filter index has to point inside the weight table array*/
      REQUIRE( net->weight_table_size() > static_cast<sint32>(net->neuron_array(i).memory_filter_idx()) );

      /* Weights */
      REQUIRE( /* Every weight synapse element has to point inside the weight table array */
        net->weight_table_size() >= /* Equality is permitted here, because the interval iterates from (start) to (start + size - 1) */
        static_cast<sint32>( net->neuron_array(i).input_weights(weight_syn_iterator).starts()
          + net->neuron_array(i).input_weights(weight_syn_iterator).interval_size() )
      );
      for(uint32 weight_iterator = 0; weight_iterator < net->neuron_array(i).input_weights(weight_syn_iterator).interval_size(); ++weight_iterator){
        CHECK( /* The weights of the Neuron has to be inbetween (-1;1) */
          -double_literal(1.0) <= net->weight_table(
            net->neuron_array(i).input_weights(weight_syn_iterator).starts() + weight_iterator
          )
        );
        CHECK(
          double_literal(1.0) >= net->weight_table(
            net->neuron_array(i).input_weights(weight_syn_iterator).starts() + weight_iterator
          )
        );
      }
      number_of_input_weights += net->neuron_array(i).input_weights(weight_syn_iterator).interval_size();
    }

    /* See if number on inputs are the same for indexes and weights */
    CHECK( number_of_input_indexes <= number_of_input_weights );
  }

  /* Check Input neurons */
  /* Input Neurons should have 2 weight Synapses: inputs and bias */
  CHECK( 2 == net->neuron_array(0).input_weights_size() );
  CHECK( 2 == net->neuron_array(1).input_weights_size() );

  /* Input Neurons should have their first synapse starting from the 0th input */
  CHECK( Synapse_iterator<>::synapse_index_from_input_index(0) == net->neuron_array(0).input_indices(0).starts() ); /* 0th Input, translated using Synapse_iterator<> */
  CHECK( Synapse_iterator<>::synapse_index_from_input_index(0) == net->neuron_array(1).input_indices(0).starts() );

  /* The input Layer should have Identity transfer function according to configuration */
  CHECK( transfer_function_identity == net->neuron_array(0).transfer_function_idx() );
  CHECK( transfer_function_identity == net->neuron_array(1).transfer_function_idx() );

  /* Check Hidden Neurons */
  /* Hidden Neurons should have 2 weight Synapses: inputs and bias */
  CHECK( 2 == net->neuron_array(2).input_weights_size() );
  CHECK( 2 == net->neuron_array(3).input_weights_size() );
  CHECK( 2 == net->neuron_array(4).input_weights_size() );

  /* Input Neurons should have their first synapse starting from 0 as well */
  CHECK( 0 == net->neuron_array(2).input_indices(0).starts() ); /* 0th Neuron, because neuron index >= net->input_neuron_number() */
  CHECK( 0 == net->neuron_array(3).input_indices(0).starts() );
  CHECK( 0 == net->neuron_array(4).input_indices(0).starts() );

  /* The Hidden Layer should have either transfer_function_relu or transfer_function_selu according to the configuration */
  CHECK((
    (transfer_function_relu == net->neuron_array(2).transfer_function_idx())
    ||(transfer_function_selu == net->neuron_array(2).transfer_function_idx())
  ));

  CHECK((
    (transfer_function_relu == net->neuron_array(3).transfer_function_idx())
    ||(transfer_function_selu == net->neuron_array(3).transfer_function_idx())
  ));

  CHECK((
    (transfer_function_relu == net->neuron_array(4).transfer_function_idx())
    ||(transfer_function_selu == net->neuron_array(4).transfer_function_idx())
  ));

  /* Check Output Neurons */
  /* Output Neurons should have 2 input weight synapses */
  CHECK( 2 == net->neuron_array(5).input_weights_size() );
  CHECK( 2 == net->neuron_array(6).input_weights_size() );

  /* Output Neurons should have should have their synapse start at the 2nd Neuron (Previous layer start) */
  CHECK( 2 == net->neuron_array(5).input_indices(0).starts() );
  CHECK( 2 == net->neuron_array(6).input_indices(0).starts() );

  /* The Output Layer should have either transfer_function_sigmoid or transfer_function_tanh according to the configuration */
  CHECK((
    (transfer_function_sigmoid == net->neuron_array(5).transfer_function_idx())
    ||(transfer_function_tanh == net->neuron_array(5).transfer_function_idx())
  ));

  CHECK((
    (transfer_function_sigmoid == net->neuron_array(6).transfer_function_idx())
    ||(transfer_function_tanh == net->neuron_array(6).transfer_function_idx())
  ));
  return net;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface", "[build][small]" ) {
  RafkoNet* net = test_net_builder_fully_connected(nullptr);
  REQUIRE( nullptr != net );
  delete net;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface with arena", "[build][arena][small]" ) {
  google::protobuf::Arena arena;
  RafkoNet* net(test_net_builder_fully_connected(&arena));
  REQUIRE( nullptr != net );
  arena.Reset();
}

} /* namespace rafko_net_test */
