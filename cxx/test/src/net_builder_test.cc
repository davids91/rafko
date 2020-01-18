#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "models/neuron_info.h"
#include "services/sparse_net_builder.h"
#include "services/synapse_iterator.h"

#include <memory>

namespace sparse_net_library_test {

  using std::shared_ptr;
  using std::unique_ptr;
  using std::make_unique;
  using std::vector;

  using sparse_net_library::uint32;
  using sparse_net_library::sdouble32;
  using sparse_net_library::Neuron;
  using sparse_net_library::SparseNet;
  using sparse_net_library::transfer_functions;
  using sparse_net_library::Sparse_net_builder;
  using sparse_net_library::Neuron_info;
  using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
  using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
  using sparse_net_library::TRANSFER_FUNCTION_TANH;
  using sparse_net_library::TRANSFER_FUNCTION_RELU;
  using sparse_net_library::TRANSFER_FUNCTION_SELU;
  using sparse_net_library::Synapse_iterator;
  using sparse_net_library::Synapse_interval;
  using sparse_net_library::COST_FUNCTION_QUADRATIC;

/*###############################################################################################
 * Testing Manual Net creation
 * Create 3 Neurons, each having the same weight
 * The 0th Neuron shall have the input, which is a number
 * the last 2 shall be the outputs
 * 0th Neuron shall have 5 inputs
 * 1st and 2nd neurons will have the first as input both
 * */
SparseNet* test_net_builder_manually(google::protobuf::Arena* arena){

  using std::make_shared;

  /* Create the single Weight Table */
  Synapse_interval temp_synapse_interval;
  sdouble32 used_weight = 0.5;
  transfer_functions used_transfer_function = TRANSFER_FUNCTION_SIGMOID;
  vector<sdouble32> weight_table {0.0};
  weight_table[0] = used_weight;
  REQUIRE( nullptr != &(weight_table[0]) );
  REQUIRE( weight_table[0] == used_weight);

  /* Create the Neuron Table */
  vector<Neuron> neuron_table(3);

  /* Neuron 0 Has an input of 1 */
  neuron_table[0].set_transfer_function_idx(used_transfer_function);
  neuron_table[0].set_memory_filter_idx(0); /* Weight 0 in the weight_table */
  neuron_table[0].set_bias_idx(0); /* Weight 0 in the weight_table */
  temp_synapse_interval.set_starts(0); /* Input Starting from 0 */
  temp_synapse_interval.set_interval_size(1); /* 1 Input */
  *neuron_table[0].add_input_indices() = temp_synapse_interval;
  temp_synapse_interval.set_starts(0); /* Weight 0 in the weight_table */
  temp_synapse_interval.set_interval_size(1); /* only Weight 0 in the weight_table */
  *neuron_table[0].add_input_weights() = temp_synapse_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid(neuron_table[0]) );

  /* Neuron 1 Has Neuron 0 as input */
  neuron_table[1].set_transfer_function_idx(used_transfer_function);
  neuron_table[1].set_memory_filter_idx(0);
  neuron_table[1].set_bias_idx(0);
  temp_synapse_interval.set_starts(0);
  temp_synapse_interval.set_interval_size(1);
  *neuron_table[1].add_input_indices() = temp_synapse_interval;
  temp_synapse_interval.set_starts(0);
  temp_synapse_interval.set_interval_size(1);
  *neuron_table[1].add_input_weights() = temp_synapse_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid(neuron_table[1]) );

  /* Neuron 2 Also has Neuron 0 as input */
  neuron_table[2].set_transfer_function_idx(used_transfer_function);
  neuron_table[2].set_memory_filter_idx(0);
  neuron_table[2].set_bias_idx(0);
  temp_synapse_interval.set_starts(0);
  temp_synapse_interval.set_interval_size(1);
  *neuron_table[2].add_input_indices() = temp_synapse_interval;
  temp_synapse_interval.set_starts(0);
  temp_synapse_interval.set_interval_size(1);
  *neuron_table[2].add_input_weights() = temp_synapse_interval;
  REQUIRE( true == Neuron_info::is_neuron_valid(neuron_table[2]) );

  /* Pass the net into the builder */
  shared_ptr<Sparse_net_builder> builder(make_shared<Sparse_net_builder>());
  builder->input_size(1)
    .expected_input_range(1.0)
    .output_neuron_number(2)
    .arena_ptr(arena)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .neuron_array(neuron_table)
    .weight_table(weight_table);
  try{
    /* Build the net with the given parameters */
    SparseNet* net(builder->build());

    /* Check Net parameters */
    REQUIRE( 0 < net->neuron_array_size() );
    REQUIRE( 0 < net->weight_table_size() );
    CHECK( 3 == net->neuron_array_size() );
    CHECK( 1 == net->weight_table_size() );
    CHECK( used_weight == net->weight_table(0) );

    /* Check parameters for each neuron */
    REQUIRE( true == Neuron_info::is_neuron_valid(net->neuron_array(0)) );
    REQUIRE( 0 < net->neuron_array(0).input_indices_size() );
    CHECK( 1 == net->neuron_array(0).input_indices_size() );
    CHECK( 1 == net->neuron_array(0).input_indices(0).interval_size() );
    CHECK( 0 == net->neuron_array(0).input_indices(0).starts() );
    REQUIRE( 0 < net->neuron_array(0).input_weights_size() );
    CHECK( 1 == net->neuron_array(0).input_weights_size() );
    CHECK( 1 == net->neuron_array(0).input_weights(0).interval_size() );
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
    CHECK( 1 == net->neuron_array(1).input_weights(0).interval_size() );
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
    CHECK( 1 == net->neuron_array(2).input_weights(0).interval_size() );
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
  SparseNet* net = test_net_builder_manually(nullptr);
  REQUIRE( nullptr != net );
  delete net;
}

TEST_CASE("Constructing small net manually using arena","[build][arena][small][manual]"){
  google::protobuf::Arena arena;
  SparseNet* net = test_net_builder_manually(&arena);
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
SparseNet* test_net_builder_fully_connected(google::protobuf::Arena* arena){
  unique_ptr<Sparse_net_builder> builder(make_unique<Sparse_net_builder>());
  builder->input_size(5)
    .output_neuron_number(2)
    .expected_input_range(5.0)
    .cost_function(COST_FUNCTION_QUADRATIC)
    .arena_ptr(arena);

  SparseNet* net(builder->dense_layers(
    {2,3,2},{
      {TRANSFER_FUNCTION_IDENTITY},
      {TRANSFER_FUNCTION_SELU,TRANSFER_FUNCTION_RELU},
      {TRANSFER_FUNCTION_TANH,TRANSFER_FUNCTION_SIGMOID}
    }
  ));

  /* Check net validity in general */
  REQUIRE( 0 < net->weight_table_size() );
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
    for(int index_synapse_iterator = 0; index_synapse_iterator < net->neuron_array(i).input_indices_size(); ++index_synapse_iterator){
      REQUIRE( /* Every index synapse element has to point inside the neuron array */
        net->neuron_array_size() >
        ( net->neuron_array(i).input_indices(index_synapse_iterator).starts()
          + net->neuron_array(i).input_indices(index_synapse_iterator).interval_size() )
      );
      number_of_input_indexes += net->neuron_array(i).input_indices(index_synapse_iterator).interval_size();
    }


    /* Check Weight indexes */
    number_of_input_weights = 0;
    REQUIRE( 0 < net->neuron_array(i).input_weights_size() );
    for(int weight_synapse_iterator = 0; weight_synapse_iterator < net->neuron_array(i).input_weights_size(); ++weight_synapse_iterator){
      /* Bias and memory filter index has to point inside the weight table array*/
      REQUIRE( net->weight_table_size() > net->neuron_array(i).bias_idx() );
      REQUIRE( net->weight_table_size() > net->neuron_array(i).memory_filter_idx() );

      /* Weights */
      REQUIRE( /* Every weight synapse element has to point inside the weight table array */
        net->weight_table_size() >
        ( net->neuron_array(i).input_weights(weight_synapse_iterator).starts()
          + net->neuron_array(i).input_weights(weight_synapse_iterator).interval_size() )
      );
      for(uint32 weight_iterator = 0; weight_iterator < net->neuron_array(i).input_weights(weight_synapse_iterator).interval_size(); ++weight_iterator){
        CHECK( /* The weights of the Neuron has to be inbetween (-1;1) */
          -1.0 <= net->weight_table(
            net->neuron_array(i).input_weights(weight_synapse_iterator).starts() + weight_iterator
          )
        );
        CHECK(
          1.0 >= net->weight_table(
            net->neuron_array(i).input_weights(weight_synapse_iterator).starts() + weight_iterator
          )
        );
      }
      number_of_input_weights += net->neuron_array(i).input_weights(weight_synapse_iterator).interval_size();
    }

    /* See if number on inputs are the same for indexes and weights */
    CHECK( number_of_input_indexes == number_of_input_weights );
  }

  /* Check Input neurons */
  /* Input Neurons should have 1 weight Synapse */
  CHECK( 1 == net->neuron_array(0).input_weights_size() );
  CHECK( 1 == net->neuron_array(1).input_weights_size() );

  /* Input Neurons should have their synapse starting from the 0th input */
  CHECK( Synapse_iterator::synapse_index_from_input_index(0) == net->neuron_array(0).input_indices(0).starts() ); /* 0th Input, translated using Synapse_iterator */
  CHECK( Synapse_iterator::synapse_index_from_input_index(0) == net->neuron_array(1).input_indices(0).starts() );

  /* The input Layer should have Identity transfer function according to configuration */
  CHECK( TRANSFER_FUNCTION_IDENTITY == net->neuron_array(0).transfer_function_idx() );
  CHECK( TRANSFER_FUNCTION_IDENTITY == net->neuron_array(1).transfer_function_idx() );

  /* Check Hidden Neurons */
  /* Hidden Neurons should have 1 weight synapse */
  CHECK( 1 == net->neuron_array(2).input_weights_size() );
  CHECK( 1 == net->neuron_array(3).input_weights_size() );
  CHECK( 1 == net->neuron_array(4).input_weights_size() );

  /* Input Neurons should have their synapse starting from 0 as well */
  CHECK( 0 == net->neuron_array(2).input_indices(0).starts() ); /* 0th Neuron, because neuron index >= net->input_neuron_number() */
  CHECK( 0 == net->neuron_array(3).input_indices(0).starts() );
  CHECK( 0 == net->neuron_array(4).input_indices(0).starts() );

  /* The Hidden Layer should have either TRANSFER_FUNCTION_RELU or TRANSFER_FUNCTION_SELU according to the configuration */
  CHECK((
    (TRANSFER_FUNCTION_RELU == net->neuron_array(2).transfer_function_idx())
    ||(TRANSFER_FUNCTION_SELU == net->neuron_array(2).transfer_function_idx())
  ));

  CHECK((
    (TRANSFER_FUNCTION_RELU == net->neuron_array(3).transfer_function_idx())
    ||(TRANSFER_FUNCTION_SELU == net->neuron_array(3).transfer_function_idx())
  ));

  CHECK((
    (TRANSFER_FUNCTION_RELU == net->neuron_array(4).transfer_function_idx())
    ||(TRANSFER_FUNCTION_SELU == net->neuron_array(4).transfer_function_idx())
  ));

  /* Check Output Neurons */
  /* Output Neurons should have 3 input weights */
  CHECK( 1 == net->neuron_array(5).input_weights_size() );
  CHECK( 1 == net->neuron_array(6).input_weights_size() );

  /* Output Neurons should have should have their synapse start at the 2nd Neuron (Previous layer start) */
  CHECK( 2 == net->neuron_array(5).input_indices(0).starts() );
  CHECK( 2 == net->neuron_array(6).input_indices(0).starts() );

  /* The Output Layer should have either TRANSFER_FUNCTION_SIGMOID or TRANSFER_FUNCTION_TANH according to the configuration */
  CHECK((
    (TRANSFER_FUNCTION_SIGMOID == net->neuron_array(5).transfer_function_idx())
    ||(TRANSFER_FUNCTION_TANH == net->neuron_array(5).transfer_function_idx())
  ));

  CHECK((
    (TRANSFER_FUNCTION_SIGMOID == net->neuron_array(6).transfer_function_idx())
    ||(TRANSFER_FUNCTION_TANH == net->neuron_array(6).transfer_function_idx())
  ));
  return net;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface", "[build][small]" ) {
  SparseNet* net = test_net_builder_fully_connected(nullptr);
  REQUIRE( nullptr != net );
  delete net;
}

TEST_CASE( "Builder to construct Fully Connected Net correctly through the interface with arena", "[build][arena][small]" ) {
  google::protobuf::Arena arena;
  SparseNet* net(test_net_builder_fully_connected(&arena));
  REQUIRE( nullptr != net );
  arena.Reset();
}

} /* namespace sparse_net_library_test */
