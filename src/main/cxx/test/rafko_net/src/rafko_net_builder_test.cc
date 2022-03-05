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
#include <tuple>
#include <vector>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/models/input_function.h"
#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/synapse_iterator.h"

#include "test/test_utility.h"

namespace rafko_net_test {

/*###############################################################################################
 * Testing Manual Net creation
 * Create 3 Neurons, each having the same weight
 * The 0th Neuron shall have the input, which is a number
 * the last 2 shall be the outputs
 * 0th Neuron shall have 5 inputs
 * 1st and 2nd neurons will have the first as input both
 * */
rafko_net::RafkoNet* test_net_builder_manually(google::protobuf::Arena* arena){
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings().set_arena_ptr(arena);

  /* Create the single Weight Table */
  rafko_net::IndexSynapseInterval temp_index_interval;
  rafko_net::InputSynapseInterval temp_input_interval;
  double used_weight = (0.5);
  rafko_net::Transfer_functions used_transfer_function = rafko_net::transfer_function_sigmoid;
  std::vector<double> weight_table {(0.0),(0.0),(0.0)};
  weight_table[0] = used_weight;
  REQUIRE( nullptr != &(weight_table[0]) );
  REQUIRE( weight_table[0] == used_weight);

  /* Create the Neuron Table */
  std::vector<rafko_net::Neuron> neuron_table(3);

  /* Neuron 0 Has an input of 1 */
  neuron_table[0].set_transfer_function(used_transfer_function);
  temp_input_interval.set_starts(0); /* Input Starting from 0 */
  temp_input_interval.set_interval_size(1); /* 1 Input */
  *neuron_table[0].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0); /* Weight 0 in the weight_table */
  temp_index_interval.set_interval_size(3); /* Spike function weight + Weight0 + bias0 in the weight_table */
  *neuron_table[0].add_input_weights() = temp_index_interval;
  REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid( neuron_table[0]) );

  /* Neuron 1 Has Neuron 0 as input */
  neuron_table[1].set_transfer_function(used_transfer_function);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(1);
  *neuron_table[1].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(3);
  *neuron_table[1].add_input_weights() = temp_index_interval;
  REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid( neuron_table[1]) );

  /* Neuron 2 Also has Neuron 0 as input */
  neuron_table[2].set_transfer_function(used_transfer_function);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(1);
  *neuron_table[2].add_input_indices() = temp_input_interval;
  temp_index_interval.set_starts(0);
  temp_index_interval.set_interval_size(3);
  *neuron_table[2].add_input_weights() = temp_index_interval;
  REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid( neuron_table[2]) );

  /* Pass the net into the builder */
  std::shared_ptr<rafko_net::RafkoNetBuilder> builder(std::make_shared<rafko_net::RafkoNetBuilder>(settings));
  builder->input_size(1).expected_input_range((1.0)).output_neuron_number(2)
    .neuron_array(neuron_table).weight_table(weight_table);

  try{
    /* Build the net with the given parameters */
    rafko_net::RafkoNet* network(builder->build());

    /* Check Net parameters */
    REQUIRE( 0 < network->neuron_array_size() );
    REQUIRE( 0 < network->weight_table_size() );
    CHECK( 3 == network->neuron_array_size() );
    CHECK( 3 == network->weight_table_size() );
    CHECK( used_weight == network->weight_table(0) );

    /* Check parameters for each neuron */
    REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid(network->neuron_array(0)) );
    REQUIRE( 0 < network->neuron_array(0).input_indices_size() );
    CHECK( 1 == network->neuron_array(0).input_indices_size() );
    CHECK( 1 == network->neuron_array(0).input_indices(0).interval_size() );
    CHECK( 0 == network->neuron_array(0).input_indices(0).starts() );
    REQUIRE( 0 < network->neuron_array(0).input_weights_size() );
    CHECK( 1 == network->neuron_array(0).input_weights_size() );
    CHECK( 3 == network->neuron_array(0).input_weights(0).interval_size() );
    CHECK( 0 == network->neuron_array(0).input_weights(0).starts() );
    CHECK(
      weight_table[network->neuron_array(0).input_weights(0).starts()]
      == network->weight_table(network->neuron_array(0).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == network->neuron_array(0).transfer_function()
    );

    REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid(network->neuron_array(1)) );
    REQUIRE( 0 < network->neuron_array(1).input_indices_size() );
    CHECK( 1 == network->neuron_array(1).input_indices_size() );
    CHECK( 1 == network->neuron_array(1).input_indices(0).interval_size() );
    CHECK( 0 == network->neuron_array(1).input_indices(0).starts() );
    REQUIRE( 0 < network->neuron_array(1).input_weights_size() );
    CHECK( 1 == network->neuron_array(1).input_weights_size() );
    CHECK( 3 == network->neuron_array(1).input_weights(0).interval_size() );
    CHECK( 0 == network->neuron_array(1).input_weights(0).starts() );
    CHECK(
      weight_table[network->neuron_array(1).input_weights(0).starts()]
      == network->weight_table(network->neuron_array(1).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == network->neuron_array(1).transfer_function()
    );

    REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid(network->neuron_array(2)) );
    REQUIRE( 0 < network->neuron_array(2).input_indices_size() );
    CHECK( 1 == network->neuron_array(2).input_indices_size() );
    CHECK( 1 == network->neuron_array(2).input_indices(0).interval_size() );
    CHECK( 0 == network->neuron_array(2).input_indices(0).starts() );
    REQUIRE( 0 < network->neuron_array(2).input_weights_size() );
    CHECK( 3 == network->neuron_array(2).input_weights(0).interval_size() );
    CHECK( 0 == network->neuron_array(2).input_weights(0).starts() );
    CHECK(
      weight_table[network->neuron_array(2).input_weights(0).starts()]
      == network->weight_table(network->neuron_array(2).input_weights(0).starts())
    );
    CHECK(
      used_transfer_function
      == network->neuron_array(2).transfer_function()
    );
    return network;
  }catch(const char* e){
    return nullptr;
  }
}

TEST_CASE( "Constructing small net manually", "[build][small][manual]" ) {
  rafko_net::RafkoNet* network = test_net_builder_manually(nullptr);
  REQUIRE( nullptr != network );
  delete network;
}

TEST_CASE("Constructing small net manually using arena","[build][arena][small][manual]"){
  google::protobuf::Arena arena;
  rafko_net::RafkoNet* network = test_net_builder_manually(&arena);
  REQUIRE( nullptr != network );
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
rafko_net::RafkoNet* test_net_builder_fully_connected(google::protobuf::Arena* arena){
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings().set_arena_ptr(arena);
  std::unique_ptr<rafko_net::RafkoNetBuilder> builder = std::make_unique<rafko_net::RafkoNetBuilder>(settings);
  builder->input_size(5)
    .output_neuron_number(2)
    .expected_input_range((5.0));

  rafko_net::RafkoNet* network(builder->dense_layers(
    {2,3,2},{
      {rafko_net::transfer_function_identity},
      {rafko_net::transfer_function_selu,rafko_net::transfer_function_relu},
      {rafko_net::transfer_function_tanh,rafko_net::transfer_function_sigmoid}
    }
  ));

  /* Check net validity in general */
  REQUIRE( ((5*2+2*2) + (2*3+3*2) + (2*3+2*2)) == network->weight_table_size() );
  REQUIRE( 0 < network->neuron_array_size() );
  CHECK( 7 == network->neuron_array_size() );
  CHECK( 5 == network->input_data_size() );
  CHECK( 2 == network->output_neuron_number() );

  /* Check Neuron validity in general */
  std::uint32_t number_of_input_indexes;
  std::uint32_t number_of_input_weights;
  for(int i = 0; i < 7; ++i){
    REQUIRE( true == rafko_net::NeuronInfo::is_neuron_valid(network->neuron_array(i)) );

    /* Check the indexing */
    REQUIRE( 0 < network->neuron_array(i).input_indices_size() );
    CHECK( 1 == network->neuron_array(i).input_indices_size() ); /* One synapse ==> the previous layer */
    number_of_input_indexes = 0;
    for(std::int32_t index_syn_iter = 0; index_syn_iter < network->neuron_array(i).input_indices_size(); ++index_syn_iter){
      REQUIRE( /* Every index synapse element has to point inside the neuron array */
        network->neuron_array_size() >
        static_cast<std::int32_t>( network->neuron_array(i).input_indices(index_syn_iter).starts()
          + network->neuron_array(i).input_indices(index_syn_iter).interval_size() )
      );
      number_of_input_indexes += network->neuron_array(i).input_indices(index_syn_iter).interval_size();
    }


    /* Check Weight indexes */
    number_of_input_weights = 0;
    REQUIRE( 0 < network->neuron_array(i).input_weights_size() );
    for(std::int32_t weight_syn_iterator = 0; weight_syn_iterator < network->neuron_array(i).input_weights_size(); ++weight_syn_iterator){
      /* Weights */
      REQUIRE( /* Every weight synapse element has to point inside the weight table array */
        network->weight_table_size() >= /* Equality is permitted here, because the interval iterates from (start) to (start + size - 1) */
        static_cast<std::int32_t>( network->neuron_array(i).input_weights(weight_syn_iterator).starts()
          + network->neuron_array(i).input_weights(weight_syn_iterator).interval_size() )
      );
      for(std::uint32_t weight_iterator = 0; weight_iterator < network->neuron_array(i).input_weights(weight_syn_iterator).interval_size(); ++weight_iterator){
        CHECK( /* The weights of the Neuron has to be inbetween (-1;1) */
          -(1.0) <= network->weight_table(
            network->neuron_array(i).input_weights(weight_syn_iterator).starts() + weight_iterator
          )
        );
        CHECK(
          (1.0) >= network->weight_table(
            network->neuron_array(i).input_weights(weight_syn_iterator).starts() + weight_iterator
          )
        );
      }
      number_of_input_weights += network->neuron_array(i).input_weights(weight_syn_iterator).interval_size();
    }

    /* See if number on inputs are the same for indexes and weights */
    CHECK( number_of_input_indexes <= number_of_input_weights );
  }

  /* Check Input neurons */
  /* Input Neurons should have 1 weight Synapse for the inputs and a bias */
  CHECK( 1 == network->neuron_array(0).input_weights_size() );
  CHECK( 1 == network->neuron_array(1).input_weights_size() );

  /* Input Neurons should have their first synapse starting from the 0th input */
  CHECK( rafko_net::SynapseIterator<>::synapse_index_from_input_index(0) == network->neuron_array(0).input_indices(0).starts() ); /* 0th Input, translated using SynapseIterator<> */
  CHECK( rafko_net::SynapseIterator<>::synapse_index_from_input_index(0) == network->neuron_array(1).input_indices(0).starts() );

  /* The input Layer should have Identity transfer function according to configuration */
  CHECK( rafko_net::transfer_function_identity == network->neuron_array(0).transfer_function() );
  CHECK( rafko_net::transfer_function_identity == network->neuron_array(1).transfer_function() );

  /* Check Hidden Neurons */
  /* Hidden Neurons should have 1 weight Synapse for the inputs and bias */
  CHECK( 1 == network->neuron_array(2).input_weights_size() );
  CHECK( 1 == network->neuron_array(3).input_weights_size() );
  CHECK( 1 == network->neuron_array(4).input_weights_size() );

  /* Input Neurons should have their first synapse starting from 0 as well */
  CHECK( 0 == network->neuron_array(2).input_indices(0).starts() ); /* 0th Neuron, because neuron index >= network->input_neuron_number() */
  CHECK( 0 == network->neuron_array(3).input_indices(0).starts() );
  CHECK( 0 == network->neuron_array(4).input_indices(0).starts() );

  /* The Hidden Layer should have either transfer_function_relu or transfer_function_selu according to the configuration */
  CHECK((
    (rafko_net::transfer_function_relu == network->neuron_array(2).transfer_function())
    ||(rafko_net::transfer_function_selu == network->neuron_array(2).transfer_function())
  ));

  CHECK((
    (rafko_net::transfer_function_relu == network->neuron_array(3).transfer_function())
    ||(rafko_net::transfer_function_selu == network->neuron_array(3).transfer_function())
  ));

  CHECK((
    (rafko_net::transfer_function_relu == network->neuron_array(4).transfer_function())
    ||(rafko_net::transfer_function_selu == network->neuron_array(4).transfer_function())
  ));

  /* Check Output Neurons */
  /* Output Neurons should have 1 input weight synapse */
  CHECK( 1 == network->neuron_array(5).input_weights_size() );
  CHECK( 1 == network->neuron_array(6).input_weights_size() );

  /* Output Neurons should their synapse start at the 2nd Neuron (Previous layer start) */
  CHECK( 2 == network->neuron_array(5).input_indices(0).starts() );
  CHECK( 2 == network->neuron_array(6).input_indices(0).starts() );

  /* The Output Layer should have either transfer_function_sigmoid or transfer_function_tanh according to the configuration */
  CHECK((
    (rafko_net::transfer_function_sigmoid == network->neuron_array(5).transfer_function())
    ||(rafko_net::transfer_function_tanh == network->neuron_array(5).transfer_function())
  ));

  CHECK((
    (rafko_net::transfer_function_sigmoid == network->neuron_array(6).transfer_function())
    ||(rafko_net::transfer_function_tanh == network->neuron_array(6).transfer_function())
  ));
  return network;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface", "[build][small]" ) {
  rafko_net::RafkoNet* network = test_net_builder_fully_connected(nullptr);
  REQUIRE( nullptr != network);
  delete network;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface with arena", "[build][arena][small]" ) {
  google::protobuf::Arena arena;
  rafko_net::RafkoNet* network(test_net_builder_fully_connected(&arena));
  REQUIRE( nullptr != network);
  arena.Reset();
}

TEST_CASE( "Testing builder for setting Neuron input functions manually", "[build][input-function]" ) {
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings().set_arena_ptr(&arena);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder(settings);

    std::vector<std::uint32_t> net_structure;
    while((rand()%10 < 9)||(4 > net_structure.size()))
      net_structure.push_back(static_cast<std::uint32_t>(rand()%5) + 1u);

    builder.input_size(2)
      .output_neuron_number(net_structure.back())
      .expected_input_range((5.0));

    std::vector<std::tuple<std::uint32_t,std::uint32_t,rafko_net::Input_functions>> set_neuron_input_functions;
    for(std::uint32_t layer_index = 0u; layer_index < net_structure.size(); ++layer_index){
      for(std::uint32_t tries = 0; tries < 5u; ++tries){
        std::uint32_t layer_neuron_index = rand()%(net_structure[layer_index]);
        std::tuple<std::uint32_t,std::uint32_t,rafko_net::Input_functions> new_item = std::make_tuple(
          layer_index, layer_neuron_index,
          rafko_net::InputFunction::next(rafko_net::InputFunction::all_input_functions)
        );
        if( /* only add the the reference vector if its not present yet in the same layer_index/neuron_index */
          std::find_if(
            set_neuron_input_functions.begin(),set_neuron_input_functions.end(),
            [&new_item](const std::tuple<std::uint32_t,std::uint32_t,rafko_net::Input_functions>& current_item){
              return ( (std::get<0>(new_item) == std::get<0>(current_item))&&(std::get<1>(new_item) == std::get<1>(current_item)) );
            }
          ) == set_neuron_input_functions.end()
        ){
          builder.set_neuron_input_function(layer_index, layer_neuron_index, std::get<2>(new_item));
          set_neuron_input_functions.push_back( std::move(new_item) );
        }
      }/*for(5 tries)*/
    }

    rafko_net::RafkoNet* network(builder.dense_layers(net_structure));
    std::vector<std::uint32_t> layer_starts(net_structure.size());
    std::uint32_t layer_start_iterator = 0u;
    for(std::uint32_t layer_index = 0u; layer_index < net_structure.size(); ++layer_index){
      layer_starts[layer_index] = layer_start_iterator;
      layer_start_iterator += net_structure[layer_index];
    }
    for(const std::tuple<std::uint32_t,std::uint32_t,rafko_net::Input_functions>& element : set_neuron_input_functions){
      REQUIRE( std::get<0>(element) < layer_starts.size() );
      REQUIRE( std::get<1>(element) < net_structure[std::get<0>(element)] );
      std::uint32_t neuron_index = layer_starts[std::get<0>(element)] + std::get<1>(element);

      REQUIRE( network->neuron_array(neuron_index).input_function() == std::get<2>(element) );
    }
  }/*for(10 variants)*/
}

TEST_CASE( "Testing builder for setting Neuron spike functions manually", "[build][spike-function]" ) {
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings().set_arena_ptr(&arena);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder(settings);

    std::vector<std::uint32_t> net_structure;
    while((rand()%10 < 9)||(4 > net_structure.size()))
      net_structure.push_back(static_cast<std::uint32_t>(rand()%5) + 1u);

    builder.input_size(2)
      .output_neuron_number(net_structure.back())
      .expected_input_range((5.0));

    std::vector<std::tuple<std::uint32_t,std::uint32_t,rafko_net::Spike_functions>> set_neuron_spike_functions;
    for(std::uint32_t layer_index = 0u; layer_index < net_structure.size(); ++layer_index){
      for(std::uint32_t tries = 0; tries < 5u; ++tries){
        std::uint32_t layer_neuron_index = rand()%(net_structure[layer_index]);
        std::tuple<std::uint32_t,std::uint32_t,rafko_net::Spike_functions> new_item = std::make_tuple(
          layer_index, layer_neuron_index,
          rafko_net::SpikeFunction::next(rafko_net::SpikeFunction::all_spike_functions)
        );
        if( /* only add the the reference vector if its not present yet in the same layer_index/neuron_index */
          std::find_if(
            set_neuron_spike_functions.begin(),set_neuron_spike_functions.end(),
            [&new_item](const std::tuple<std::uint32_t,std::uint32_t,rafko_net::Spike_functions>& current_item){
              return ( (std::get<0>(new_item) == std::get<0>(current_item))&&(std::get<1>(new_item) == std::get<1>(current_item)) );
            }
          ) == set_neuron_spike_functions.end()
        ){
          builder.set_neuron_spike_function(layer_index, layer_neuron_index, std::get<2>(new_item));
          set_neuron_spike_functions.push_back( std::move(new_item) );
        }
      }/*for(5 tries)*/
    }

    rafko_net::RafkoNet* network(builder.dense_layers(net_structure));
    std::vector<std::uint32_t> layer_starts(net_structure.size());
    std::uint32_t layer_start_iterator = 0u;
    for(std::uint32_t layer_index = 0u; layer_index < net_structure.size(); ++layer_index){
      layer_starts[layer_index] = layer_start_iterator;
      layer_start_iterator += net_structure[layer_index];
    }
    for(const std::tuple<std::uint32_t,std::uint32_t,rafko_net::Spike_functions>& element : set_neuron_spike_functions){
      REQUIRE( std::get<0>(element) < layer_starts.size() );
      REQUIRE( std::get<1>(element) < net_structure[std::get<0>(element)] );
      std::uint32_t neuron_index = layer_starts[std::get<0>(element)] + std::get<1>(element);

      REQUIRE( network->neuron_array(neuron_index).spike_function() == std::get<2>(element) );
    }
  }/*for(10 variants)*/
}

} /* namespace rafko_net_test */
