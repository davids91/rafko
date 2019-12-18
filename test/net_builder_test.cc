#include <memory>

#include "catch.hpp"

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"
#include "services/sparse_net_builder.h"

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
  using sparse_net_library::SparseNetBuilder;
  using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
  using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
  using sparse_net_library::TRANSFER_FUNCTION_TANH;
  using sparse_net_library::TRANSFER_FUNCTION_RELU;
  using sparse_net_library::TRANSFER_FUNCTION_SELU;
  using sparse_net_library::INVALID_USAGE_EXCEPTION;
  using sparse_net_library::NOT_IMPLEMENTED_EXCEPTION;

/*###############################################################################################
 * Testing Neuron Validation
 * */
TEST_CASE( "Constructing a Neuron Manually", "[Neuron][small][manual]" ) {
  /* Empty Neuron should be invalid */
  unique_ptr<Neuron> neuron = make_unique<Neuron>();
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  /* Setting some parameters */
  neuron->set_bias_idx(0); /* Unfortunately checking against the weight table is not possible without Net context */
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->set_memory_ratio_idx(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->set_transfer_function_idx(TRANSFER_FUNCTION_IDENTITY);
  CHECK( true == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  /* Setting indexing information */
  neuron->add_weight_index_sizes(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_weight_index_starts(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_input_index_sizes(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_input_index_starts(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->set_input_index_sizes(0,5); 
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->set_weight_index_sizes(0,5); 
  CHECK( true == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_weight_index_sizes(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_weight_index_starts(0);
  CHECK( true == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_input_index_sizes(0);
  CHECK( false == SparseNetBuilder::is_neuron_valid(neuron.get()) );

  neuron->add_input_index_starts(0); /* Only the first partition size is checked for non-zero */
  CHECK( true == SparseNetBuilder::is_neuron_valid(neuron.get()) );
}

/*###############################################################################################
 * Testing Manual Net creation
 * Create 3 Neurons, each having the same weight
 * The 0th Neuron shall have the input, which is a number
 * the last 2 shall be the outputs
 * 0th Neuron shall have 5 inputs
 * 1st and 2nd neurons will have the first as input both
 * */
SparseNet* test_net_builder_manually(google::protobuf::Arena* arena){
  /* Create the single Weight Table */
  sdouble32 used_weight = 0.5;
  transfer_functions used_transfer_function = TRANSFER_FUNCTION_SIGMOID;
  vector<sdouble32> weight_table {0.0};
  weight_table[0] = used_weight;
  REQUIRE( nullptr != &(weight_table[0]) );
  REQUIRE( weight_table[0] == used_weight);

  /* Create the Neuron Table */ /*! #7 */
  vector<Neuron> neuron_table(3);

  /* Neuron 0 Has an input of 1 */
  neuron_table[0].set_transfer_function_idx(used_transfer_function);
  neuron_table[0].set_memory_ratio_idx(0); /* Weight 0 in the weight_table */
  neuron_table[0].set_bias_idx(0); /* Weight 0 in the weight_table */
  neuron_table[0].add_input_index_sizes(1); /* 1 Input */
  neuron_table[0].add_input_index_starts(0); /* Starting from 0 */
  neuron_table[0].add_weight_index_sizes(1); /* Weight 0 in the weight_table */
  neuron_table[0].add_weight_index_starts(0); /* Weight 0 in the weight_table */
  REQUIRE( true == SparseNetBuilder::is_neuron_valid(&(neuron_table[0])) );

  /* Neuron 1 Has Neuron 0 as input */
  neuron_table[1].set_transfer_function_idx(used_transfer_function);
  neuron_table[1].set_memory_ratio_idx(0);
  neuron_table[1].set_bias_idx(0);
  neuron_table[1].add_input_index_sizes(1);
  neuron_table[1].add_input_index_starts(0);
  neuron_table[1].add_weight_index_sizes(1);
  neuron_table[1].add_weight_index_starts(0);
  REQUIRE( true == SparseNetBuilder::is_neuron_valid(&(neuron_table[1])) );

  /* Neuron 2 Also has Neuron 0 as input */
  neuron_table[2].set_transfer_function_idx(used_transfer_function);
  neuron_table[2].set_memory_ratio_idx(0);
  neuron_table[2].set_bias_idx(0);
  neuron_table[2].add_input_index_sizes(1);
  neuron_table[2].add_input_index_starts(0);
  neuron_table[2].add_weight_index_sizes(1);
  neuron_table[2].add_weight_index_starts(0);
  REQUIRE( true == SparseNetBuilder::is_neuron_valid(&(neuron_table[2])) );

  /* Pass the net into the builder */
  shared_ptr<SparseNetBuilder> builder(new SparseNetBuilder());
  builder->input_size(1)
    .input_neuron_size(1)
    .expectedInputRange(1.0)
    .output_neuron_number(2)
    .arena_ptr(arena)
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
    REQUIRE( true == SparseNetBuilder::is_neuron_valid(&net->neuron_array(0)) );
    REQUIRE( 0 < net->neuron_array(0).input_index_sizes_size() );
    REQUIRE( net->neuron_array(0).input_index_sizes_size() == net->neuron_array(0).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(0).input_index_sizes_size() );
    CHECK( 1 == net->neuron_array(0).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(0).input_index_sizes(0) );
    CHECK( 0 == net->neuron_array(0).input_index_starts(0) );
    REQUIRE( 0 < net->neuron_array(0).weight_index_sizes_size() );
    REQUIRE( net->neuron_array(0).weight_index_sizes_size() == net->neuron_array(0).weight_index_starts_size() );
    CHECK( 1 == net->neuron_array(0).weight_index_sizes_size() );
    CHECK( 1 == net->neuron_array(0).weight_index_starts_size() );
    CHECK( 1 == net->neuron_array(0).weight_index_sizes(0) );
    CHECK( 0 == net->neuron_array(0).weight_index_starts(0) );
    CHECK(
      weight_table[net->neuron_array(0).weight_index_starts(0)]
      == net->weight_table(net->neuron_array(0).weight_index_starts(0))
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(0).transfer_function_idx()
    );

    REQUIRE( true == SparseNetBuilder::is_neuron_valid(&net->neuron_array(1)) );
    REQUIRE( 0 < net->neuron_array(1).input_index_sizes_size() );
    REQUIRE( net->neuron_array(1).input_index_sizes_size() == net->neuron_array(1).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(1).input_index_sizes_size() );
    CHECK( 1 == net->neuron_array(1).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(1).input_index_sizes(0) );
    CHECK( 0 == net->neuron_array(1).input_index_starts(0) );
    REQUIRE( 0 < net->neuron_array(1).weight_index_sizes_size() );
    REQUIRE( net->neuron_array(1).weight_index_sizes_size() == net->neuron_array(1).weight_index_starts_size() );
    CHECK( 1 == net->neuron_array(1).weight_index_sizes_size() );
    CHECK( 1 == net->neuron_array(1).weight_index_starts_size() );
    CHECK( 1 == net->neuron_array(1).weight_index_sizes(0) );
    CHECK( 0 == net->neuron_array(1).weight_index_starts(0) );
    CHECK(
      weight_table[net->neuron_array(1).weight_index_starts(0)]
      == net->weight_table(net->neuron_array(1).weight_index_starts(0))
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(1).transfer_function_idx()
    );

    REQUIRE( true == SparseNetBuilder::is_neuron_valid(&net->neuron_array(2)) );
    REQUIRE( 0 < net->neuron_array(2).input_index_sizes_size() );
    REQUIRE( net->neuron_array(2).input_index_sizes_size() == net->neuron_array(2).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(2).input_index_sizes_size() );
    CHECK( 1 == net->neuron_array(2).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(2).input_index_sizes(0) );
    CHECK( 0 == net->neuron_array(2).input_index_starts(0) );
    REQUIRE( net->neuron_array(2).weight_index_sizes_size() == net->neuron_array(2).weight_index_starts_size() );
    REQUIRE( 0 < net->neuron_array(2).weight_index_sizes_size() );
    CHECK( 1 == net->neuron_array(2).weight_index_sizes(0) );
    CHECK( 0 == net->neuron_array(2).weight_index_starts(0) );
    CHECK(
      weight_table[net->neuron_array(2).weight_index_starts(0)]
      == net->weight_table(net->neuron_array(2).weight_index_starts(0))
    );
    CHECK(
      used_transfer_function
      == net->neuron_array(2).transfer_function_idx()
    );
    return net;
  }catch(int e){
    switch(e){
    case INVALID_USAGE_EXCEPTION:{
      INFO("Invalid Builder usage!");
      }break;
    case NOT_IMPLEMENTED_EXCEPTION:{
      INFO("Something is not implemented inside the Builder!");
      }break;
    default:{
      INFO("Unknown Exception!" << e);
      }break;
    }
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
  unique_ptr<SparseNetBuilder> builder(new SparseNetBuilder());
  builder->input_size(5)
    .input_neuron_size(2)
    .output_neuron_number(2)
    .expectedInputRange(5.0)
    .arena_ptr(arena);

  SparseNet* net(builder->denseLayers(
    {2,3,2},
    {{TRANSFER_FUNCTION_IDENTITY},{TRANSFER_FUNCTION_SELU,TRANSFER_FUNCTION_RELU},{TRANSFER_FUNCTION_TANH,TRANSFER_FUNCTION_SIGMOID}
  }));

  /* Check net validity in general */
  REQUIRE( 0 < net->weight_table_size() );
  REQUIRE( 0 < net->neuron_array_size() );
  CHECK( 7 == net->neuron_array_size() );
  CHECK( 5 == net->input_data_size() );
  CHECK( 2 == net->input_neuron_number() );
  CHECK( 2 == net->output_neuron_number() );

  /* Check Neuron validity in general */
  uint32 number_of_input_indexes;
  uint32 number_of_input_weights;
  for(int i = 0; i < 7; ++i){
    REQUIRE( true == SparseNetBuilder::is_neuron_valid(&net->neuron_array(i)) );

    /* Check the indexing */
    REQUIRE( 0 < net->neuron_array(i).input_index_sizes_size() );
    REQUIRE( net->neuron_array(i).input_index_sizes_size() == net->neuron_array(i).input_index_starts_size() );
    CHECK( 1 == net->neuron_array(i).input_index_sizes_size() ); /* One partition ==> the previous layer */
    number_of_input_indexes = 0;
    for(int index_partition_iterator = 0; index_partition_iterator < net->neuron_array(i).input_index_sizes_size(); ++index_partition_iterator){
      REQUIRE( /* Every index Partition element has to point inside the neuron array */
        net->neuron_array_size() > 
        ( net->neuron_array(i).input_index_starts(index_partition_iterator) 
          + net->neuron_array(i).input_index_sizes(index_partition_iterator) ) 
      );
      number_of_input_indexes += net->neuron_array(i).input_index_sizes(index_partition_iterator);
    }


    /* Check Weight indexes */
    number_of_input_weights = 0;
    REQUIRE( 0 < net->neuron_array(i).weight_index_sizes_size() );
    REQUIRE( net->neuron_array(i).weight_index_sizes_size() == net->neuron_array(i).weight_index_starts_size() );
    for(int weight_partition_iterator = 0; weight_partition_iterator < net->neuron_array(i).weight_index_starts_size(); ++weight_partition_iterator){
      /* Bias and memory ratio index has to point inside the weight table array*/
      REQUIRE( net->weight_table_size() > net->neuron_array(i).bias_idx() );
      REQUIRE( net->weight_table_size() > net->neuron_array(i).memory_ratio_idx() );

      /* Weights */
      REQUIRE( /* Every weight Partition element has to point inside the weight table array */
        net->weight_table_size() > 
        ( net->neuron_array(i).weight_index_starts(weight_partition_iterator) 
          + net->neuron_array(i).weight_index_sizes(weight_partition_iterator) ) 
      );
      for(int weight_iterator = 0; weight_iterator < net->neuron_array(i).weight_index_sizes(weight_partition_iterator); ++weight_iterator){
        CHECK( /* The weights of the Neuron has to be inbetween (-1;1) */
          -1.0 <= net->weight_table(
            net->neuron_array(i).weight_index_starts(weight_partition_iterator) + weight_iterator
          ) 
        );
        CHECK( 
          1.0 >= net->weight_table(
            net->neuron_array(i).weight_index_starts(weight_partition_iterator) + weight_iterator
          ) 
        );
      }
      number_of_input_weights += net->neuron_array(i).weight_index_sizes(weight_partition_iterator);
    }

    /* See if number on inputs are the same for indexes and weights */
    CHECK( number_of_input_indexes == number_of_input_weights );
  }

  /* Check Input neurons */
  /* Input Neurons should have 5 input weights */
  CHECK( 5 == net->neuron_array(0).weight_index_sizes_size() );
  CHECK( 5 == net->neuron_array(0).weight_index_starts_size() );
  CHECK( 5 == net->neuron_array(1).weight_index_sizes_size() );
  CHECK( 5 == net->neuron_array(1).weight_index_starts_size() );

  /* Input Neurons should have their partition starting from 0 */
  CHECK( 0 == net->neuron_array(0).input_index_starts(0) ); /* 0th Input, because neuron index < net->input_neuron_number() */
  CHECK( 0 == net->neuron_array(1).input_index_starts(0) );

  /* The input Layer should have Identity transfer function according to configuration */
  CHECK( TRANSFER_FUNCTION_IDENTITY == net->neuron_array(0).transfer_function_idx() );
  CHECK( TRANSFER_FUNCTION_IDENTITY == net->neuron_array(1).transfer_function_idx() );

  /* Check Hidden Neurons */
  /* Hidden Neurons should have 2 input weights */
  CHECK( 2 == net->neuron_array(2).weight_index_sizes_size() );
  CHECK( 2 == net->neuron_array(2).weight_index_starts_size() );
  CHECK( 2 == net->neuron_array(3).weight_index_sizes_size() );
  CHECK( 2 == net->neuron_array(3).weight_index_starts_size() );
  CHECK( 2 == net->neuron_array(4).weight_index_sizes_size() );
  CHECK( 2 == net->neuron_array(4).weight_index_starts_size() );

  /* Input Neurons should have their partition starting from 0 as well */
  CHECK( 0 == net->neuron_array(2).input_index_starts(0) ); /* 0th Neuron, because neuron index >= net->input_neuron_number() */
  CHECK( 0 == net->neuron_array(3).input_index_starts(0) ); 
  CHECK( 0 == net->neuron_array(4).input_index_starts(0) ); 

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
  CHECK( 3 == net->neuron_array(5).weight_index_sizes_size() );
  CHECK( 3 == net->neuron_array(5).weight_index_starts_size() );
  CHECK( 3 == net->neuron_array(6).weight_index_sizes_size() );
  CHECK( 3 == net->neuron_array(6).weight_index_starts_size() );

  /* Output Neurons should have should have their partition start at the 2nd Neuron (Previous layer start) */
  CHECK( 2 == net->neuron_array(5).input_index_starts(0) ); 
  CHECK( 2 == net->neuron_array(6).input_index_starts(0) );

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
