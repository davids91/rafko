
#include "test/catch.hpp"

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "services/sparse_net_builder.h"
#include "services/solution_builder.h"
#include "services/solution_solver.h"

namespace sparse_net_library_test {

using std::unique_ptr;
using std::make_unique;
using std::vector;

using sparse_net_library::uint32;
using sparse_net_library::sdouble32;
using sparse_net_library::SparseNetBuilder;
using sparse_net_library::SparseNet;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution;
using sparse_net_library::Solution_solver;
using sparse_net_library::Synapse_iterator;

/*###############################################################################################
 * Testing Solution generation using the @Sparse_net_builder and the @Solution_builder
 * */
Solution* test_solution_builder_manually(google::protobuf::Arena* arena, sdouble32 device_max_megabytes){
  /* Build a net */
  vector<uint32> net_structure = {20,10,30,10,2};
  unique_ptr<SparseNetBuilder> net_builder = make_unique<SparseNetBuilder>();
  net_builder->input_size(50).output_neuron_number(2).expectedInputRange(5.0).arena_ptr(arena);
  SparseNet* net(net_builder->denseLayers(net_structure));
  net_builder.reset();

  /* Generate solution from Net */
  unique_ptr<Solution_builder> solution_builder = make_unique<Solution_builder>();
  Solution* solution = solution_builder->max_solve_threads(4).device_max_megabytes(device_max_megabytes).arena_ptr(arena).build(*net);

  /* See if every Neuron is inside the result solution */
  bool found;
  for(uint32 neuron_iterator = 0; neuron_iterator < static_cast<uint32>(std::max(0,net->neuron_array_size())); ++neuron_iterator){
    found = false;
    for(int partial_solution_iterator = 0; partial_solution_iterator < solution->partial_solutions_size(); ++partial_solution_iterator){
      for(
        uint32 internal_neuron_iterator = 0;
        internal_neuron_iterator < solution->partial_solutions(partial_solution_iterator).internal_neuron_number();
        ++internal_neuron_iterator
      ){
        if(solution->partial_solutions(partial_solution_iterator).actual_index(internal_neuron_iterator) == neuron_iterator){
          found = true;
          goto Solution_search_over; /* don't judge */
        }
      }
    }
    Solution_search_over:
    CHECK( true == found ); /* Found the Neuron index from the net in the result solution */
  }

  /* Test if the inputs of the partial in the first row only contain input indexes */
  uint32 input_synapse_offset;
  uint32 weight_synapse_offset;
  uint32 neuron_synapse_element_iterator;
  for(int neuron_iterator = 0; neuron_iterator < net->neuron_array_size(); ++neuron_iterator){ /* For the input Neurons */
    for(
      int partial_solution_iterator = 0; 
      partial_solution_iterator < solution->partial_solutions_size();
      ++partial_solution_iterator
    ){ /* Search trough the partial solutions, looking for the neuron_iterator'th Neuron */
      input_synapse_offset = 0; /* Since the Neurons are sharing their input synapses in a common array, an offset needs to be calculated */
      weight_synapse_offset = 0;

      /* Since Neurons take their inputs from the partial solution input, test iterates over it */
      Synapse_iterator partial_input_iterator(solution->partial_solutions(partial_solution_iterator).input_data());
      for( /* Skim through the inner neurons in the partial solution until the current one if found */
        uint32 inner_neuron_iterator = 0;
        inner_neuron_iterator < solution->partial_solutions(partial_solution_iterator).internal_neuron_number();
        ++inner_neuron_iterator
      ){
        if(neuron_iterator == static_cast<int>(
          solution->partial_solutions(partial_solution_iterator).actual_index(inner_neuron_iterator)
        )){ /* If the current neuron being checked is the one in the partial solution under inner_neuron_iterator */
          /* Test if all of the neurons weights are equal to the ones in the net */
          neuron_synapse_element_iterator = 0;
          /* Test iterates over the Neurons inputs, to see if they point to the network inputs */
          Synapse_iterator inner_neuron_weight_iterator(solution->partial_solutions(partial_solution_iterator).weight_indices());
          Synapse_iterator neuron_weight_iterator(net->neuron_array(neuron_iterator).input_weights());
          inner_neuron_weight_iterator.iterate([&](int input_index){ /* Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator::is_index_input s true ) */
            REQUIRE( neuron_weight_iterator.size() > neuron_synapse_element_iterator );
            CHECK( 
              solution->partial_solutions(partial_solution_iterator).weight_table(input_index) 
              == net->weight_table(neuron_weight_iterator[neuron_synapse_element_iterator]) 
            );
            ++neuron_synapse_element_iterator;
          },input_synapse_offset,solution->partial_solutions(partial_solution_iterator).index_synapse_number(inner_neuron_iterator));

          /* Test if all of the neurons inputs are are the same as the ones in the net */
          neuron_synapse_element_iterator = 0;
          /* Test iterates over the inner neurons synapse to see if it matchen the Neuron synapse */
          Synapse_iterator inner_neuron_input_iterator(solution->partial_solutions(partial_solution_iterator).inside_indices());
          Synapse_iterator neuron_input_iterator(net->neuron_array(neuron_iterator).input_indices());
          inner_neuron_input_iterator.iterate([&](int input_index){ /* Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator::is_index_input s true ) */
            REQUIRE( neuron_input_iterator.size() > neuron_synapse_element_iterator );
            if(!Synapse_iterator::is_index_input(input_index)){ /* Inner neuron takes its input internally */
              CHECK( 
                solution->partial_solutions(partial_solution_iterator).actual_index(input_index) 
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
            }else{ /* Inner Neuron takes its input from the partial solution input */
              CHECK( 
                partial_input_iterator[Synapse_iterator::input_index_from_synapse_index(input_index)] 
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
            }
            ++neuron_synapse_element_iterator;
          },input_synapse_offset,solution->partial_solutions(partial_solution_iterator).index_synapse_number(inner_neuron_iterator));

          goto Neuron_found_in_partial;
        }else{ /* neuron_iterator is not under inner_neuron_iterator in the partial solutio.. adjust synapse offsets */
          input_synapse_offset += solution->partial_solutions(partial_solution_iterator).index_synapse_number(inner_neuron_iterator);
          weight_synapse_offset += solution->partial_solutions(partial_solution_iterator).weight_synapse_number(inner_neuron_iterator);
        }
      } /* Inner Neuron loop*/
    } /* Partial solution loop */
    Neuron_found_in_partial:
    input_synapse_offset = 0; /* Dummy statement so that accursed goto works with the above label. Don't use GOTO kids! ..unless you absolutely have to! */
  } /*(uint32 neuron_iterator = 0; neuron_iterator < net_structure.front(); ++neuron_iterator)*/

  /* TODO: Test if all of the neuron is present in all of the partial solutions outputs */

  return solution;
}

TEST_CASE( "Building a solution from a net", "[build][small][build-only]" ){
  sdouble32 space_used_megabytes = 0;
  Solution* solution = test_solution_builder_manually(nullptr,2048.0);
  REQUIRE( nullptr != solution );
  space_used_megabytes = solution->SpaceUsedLong() /* Bytes *// 1024.0 /* KB *// 1024.0 /* MB */;
  delete solution;

  /* test it again, but with intentionally dividing the partial solutions by multiple numbers */
  solution = test_solution_builder_manually(nullptr,space_used_megabytes/5.0);
  REQUIRE( nullptr != solution );
  delete solution;

}

} /* namespace sparse_net_library_test */
