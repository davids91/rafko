#include <iostream>

#include "catch.hpp"

#include "sparsenet_global.h"
#include "models/sNet.pb.h"
#include "services/snetbuilder.h"



namespace sparse_net_library_test {

	using sparse_net_library::sdouble32;
	using sparse_net_library::Neuron;
	using sparse_net_library::SparseNet;
	using sparse_net_library::transfer_functions;
	using sparse_net_library::SparseNetBuilder;
	using sparse_net_library::TRANSFER_FUNC_IDENTITY;
	using sparse_net_library::TRANSFER_FUNC_SIGMOID;
	using sparse_net_library::TRANSFER_FUNC_TANH;
	using sparse_net_library::TRANSFER_FUNC_RELU;
	using sparse_net_library::TRANSFER_FUNC_SELU;
	using sparse_net_library::INVALID_BUILDER_USAGE_EXCEPTION;
	using sparse_net_library::NOT_IMPLEMENTED_EXCEPTION;

bool is_neuron_valid(Neuron const * neuron)
{
  if(nullptr != neuron){
    return (
      (transfer_functions_IsValid(neuron->transfer_function_idx())) /* Transfer Function ID is valid */
      &&(0 < neuron->input_idx_size()) /* There are input idexes */
      &&(0 < neuron->input_weight_idx_size()) /* There are some connection weights */
    );
  }else return false;
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
	transfer_functions used_transfer_function = TRANSFER_FUNC_SIGMOID;
	std::shared_ptr<sdouble32[]> weight_table(new sdouble32[1]);
	weight_table[0] = used_weight;
	REQUIRE( nullptr != &(weight_table[0]) );
	REQUIRE( weight_table[0] == used_weight);

	/* Create the Neuron Table */ /*! #7 */
	std::shared_ptr<Neuron[]> neuron_table(new Neuron[3]);

	/* Neuron 0: Has Neurons 1 and 2 as inputs */
	neuron_table[0].add_input_idx(0); 
	neuron_table[0].add_input_weight_idx(0); /* Weight 0 in the weight_table */
	neuron_table[0].set_memory_ratio_idx(0); /* Weight 0 in the weight_table */
	neuron_table[0].set_transfer_function_idx(used_transfer_function); 
	REQUIRE( true == is_neuron_valid(&(neuron_table[0])) );

	neuron_table[1].add_input_idx(0); 
	neuron_table[1].add_input_weight_idx(0);
	neuron_table[1].set_memory_ratio_idx(0);
	neuron_table[1].set_transfer_function_idx(used_transfer_function); 
	REQUIRE( true == is_neuron_valid(&(neuron_table[1])) );

	neuron_table[2].add_input_idx(0); 
	neuron_table[2].add_input_weight_idx(0); 
	neuron_table[2].set_memory_ratio_idx(0);
	neuron_table[2].set_transfer_function_idx(used_transfer_function); 
	REQUIRE( true == is_neuron_valid(&(neuron_table[2])) );

  /* Pass the net into the builder */
  std::shared_ptr<SparseNetBuilder> builder(new SparseNetBuilder());
  builder->input_size(1)
	  .input_neuron_size(1)
	  .expectedInputRange(1.0)
	  .output_neuron_number(2)
	  .arena_ptr(arena)
	  .neuron_array(neuron_table,3)
	  .weight_table(weight_table,1);
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
		REQUIRE( true == is_neuron_valid(&net->neuron_array(0)) );
		REQUIRE( 0 < net->neuron_array(0).input_idx_size() );
		CHECK( 1 == net->neuron_array(0).input_idx_size() );
		CHECK( 0 == net->neuron_array(0).input_idx(0) );
		CHECK( 0 == net->neuron_array(0).input_weight_idx(0) );
		CHECK( 
			weight_table[net->neuron_array(0).input_weight_idx(0)] 
			== net->weight_table(net->neuron_array(0).input_weight_idx(0)) 
		);
		CHECK(
			used_transfer_function 
			== net->neuron_array(0).transfer_function_idx()
		);

		REQUIRE( true == is_neuron_valid(&net->neuron_array(1)) );
		REQUIRE( 0 < net->neuron_array(1).input_idx_size() );
		CHECK( 1 == net->neuron_array(1).input_idx_size() );
		CHECK( 0 == net->neuron_array(1).input_idx(0) );
		CHECK( 0 == net->neuron_array(1).input_weight_idx(0) );
		CHECK( 
			weight_table[net->neuron_array(1).input_weight_idx(0)] 
			== net->weight_table(net->neuron_array(1).input_weight_idx(0)) 
		);
		CHECK(
			used_transfer_function 
			== net->neuron_array(1).transfer_function_idx()
		);

		REQUIRE( true == is_neuron_valid(&net->neuron_array(2)) );
		REQUIRE( 0 < net->neuron_array(2).input_idx_size() );
		CHECK( 1 == net->neuron_array(2).input_idx_size() );

		CHECK( 0 == net->neuron_array(2).input_idx(0) );
		CHECK( 0 == net->neuron_array(2).input_weight_idx(0) );
		CHECK( 
			weight_table[net->neuron_array(2).input_weight_idx(0)] 
			== net->weight_table(net->neuron_array(2).input_weight_idx(0)) 
		);
		CHECK(
			used_transfer_function 
			== net->neuron_array(2).transfer_function_idx()
		);		
	  return net;
  }catch(int e){
  	switch(e){
		case INVALID_BUILDER_USAGE_EXCEPTION:{
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

TEST_CASE( "Constructing small net manually", "[build][small]" ) {
	SparseNet* net = test_net_builder_manually(nullptr);
	REQUIRE( nullptr != net );
	delete net;
}

TEST_CASE("Constructing small net manually using arena","[build][arena][small]"){
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
	std::unique_ptr<SparseNetBuilder> builder(new SparseNetBuilder());
  builder->input_size(5)
	  .input_neuron_size(2)
	  .output_neuron_number(2)
	  .expectedInputRange(5.0)
	  .arena_ptr(arena);

  SparseNet* net(builder->denseLayers(
  	{2,3,2},
  	{{TRANSFER_FUNC_IDENTITY},{TRANSFER_FUNC_SELU,TRANSFER_FUNC_RELU},{TRANSFER_FUNC_TANH,TRANSFER_FUNC_SIGMOID}
	}));

  /* Check net validity in general */
	CHECK( 7 == net->neuron_array_size() );
	CHECK( 5 == net->input_data_size() );
	CHECK( 2 == net->input_neuron_number() );
	CHECK( 2 == net->output_neuron_number() );
	CHECK( 0 < net->weight_table_size() );

	for(int i = 0; i < 7; ++i){
		REQUIRE( true == is_neuron_valid(&net->neuron_array(i)) );

		/* See if the weights are available and valid */
		REQUIRE( 0 < net->neuron_array(i).input_weight_idx_size() );
		REQUIRE( 0 < net->neuron_array(i).input_idx_size() );
		for(int weight_iterator = 0; weight_iterator < net->neuron_array(i).input_weight_idx_size(); ++weight_iterator){
			REQUIRE( net->neuron_array(i).input_weight_idx(weight_iterator) < net->weight_table_size() );
			CHECK( 0.0 <= net->weight_table(net->neuron_array(i).input_weight_idx(weight_iterator)) );
			CHECK( 1.0 >= net->weight_table(net->neuron_array(i).input_weight_idx(weight_iterator)) );
		}
	}

	/* Check Input neurons */
	/* Input Neurons should have 5 input weights */
	CHECK( 5 == net->neuron_array(0).input_weight_idx_size() );
	CHECK( 5 == net->neuron_array(1).input_weight_idx_size() );

	/* Input Neurons should have 1 input idx - because of easy partitioning in Fully connected Neurons */
	CHECK( 1 == net->neuron_array(0).input_idx_size() );
	CHECK( 0 == net->neuron_array(0).input_idx(0) ); /* 0th Neuron, because neuron index < net->input_neuron_number() */
	CHECK( 1 == net->neuron_array(1).input_idx_size() );
	CHECK( 0 == net->neuron_array(1).input_idx(0) );

	/* The input Layer should have Identity transfer function according to configuration */
	CHECK( TRANSFER_FUNC_IDENTITY == net->neuron_array(0).transfer_function_idx() );
	CHECK( TRANSFER_FUNC_IDENTITY == net->neuron_array(1).transfer_function_idx() );

	/* Check Hidden Neurons */
	/* Hidden Neurons should have 2 input weights */
	CHECK( 2 == net->neuron_array(2).input_weight_idx_size() );
	CHECK( 2 == net->neuron_array(3).input_weight_idx_size() );
	CHECK( 2 == net->neuron_array(4).input_weight_idx_size() );

	/* Hidden Neurons should have 1 input idx - because of easy partitioning in Fully connected Neurons */
	CHECK( 1 == net->neuron_array(2).input_idx_size() );
	CHECK( 0 == net->neuron_array(2).input_idx(0) ); /* 0th Neuron, because neuron index >= net->input_neuron_number() */
	CHECK( 1 == net->neuron_array(3).input_idx_size() );
	CHECK( 0 == net->neuron_array(3).input_idx(0) );
	CHECK( 1 == net->neuron_array(4).input_idx_size() );
	CHECK( 0 == net->neuron_array(4).input_idx(0) );

	/* The Hidden Layer should have either TRANSFER_FUNC_RELU or TRANSFER_FUNC_SELU according to the configuration */
	CHECK((
		(TRANSFER_FUNC_RELU == net->neuron_array(2).transfer_function_idx())
		||(TRANSFER_FUNC_SELU == net->neuron_array(2).transfer_function_idx())
	));

	CHECK((
		(TRANSFER_FUNC_RELU == net->neuron_array(3).transfer_function_idx())
		||(TRANSFER_FUNC_SELU == net->neuron_array(3).transfer_function_idx())
	));

	CHECK((
		(TRANSFER_FUNC_RELU == net->neuron_array(4).transfer_function_idx())
		||(TRANSFER_FUNC_SELU == net->neuron_array(4).transfer_function_idx())
	));

	/* Check Output Neurons */
	/* Output Neurons should have 3 input weights */
	CHECK( 3 == net->neuron_array(5).input_weight_idx_size() );
	CHECK( 3 == net->neuron_array(6).input_weight_idx_size() );

	/* Output Neurons should have 1 input idx - because of easy partitioning in Fully connected Neurons */
	CHECK( 1 == net->neuron_array(5).input_idx_size() );
	CHECK( 2 == net->neuron_array(5).input_idx(0) ); 
	CHECK( 1 == net->neuron_array(6).input_idx_size() );
	CHECK( 2 == net->neuron_array(6).input_idx(0) );

	/* The Output Layer should have either TRANSFER_FUNC_SIGMOID or TRANSFER_FUNC_TANH according to the configuration */
	CHECK((
		(TRANSFER_FUNC_SIGMOID == net->neuron_array(5).transfer_function_idx())
		||(TRANSFER_FUNC_TANH == net->neuron_array(5).transfer_function_idx())
	));

	CHECK((
		(TRANSFER_FUNC_SIGMOID == net->neuron_array(6).transfer_function_idx())
		||(TRANSFER_FUNC_TANH == net->neuron_array(6).transfer_function_idx())
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