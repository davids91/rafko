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

#include <vector>
#include <memory>
#include <numeric>

#include "gen/solution.pb.h"
#include "gen/sparse_net.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/transfer_function.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "sparse_net_library/models/spike_function.h"
#include "sparse_net_library/services/solution_solver.h"
#include "sparse_net_library/services/partial_solution_solver.h"
#include "sparse_net_library/services/synapse_iterator.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"

namespace sparse_net_library_test{

using sparse_net_library::Sparse_net_builder;
using sparse_net_library::Solution_builder;
using sparse_net_library::SparseNet;
using rafko_utilities::DataRingbuffer;
using sparse_net_library::Partial_solution;
using sparse_net_library::Partial_solution_solver;
using sparse_net_library::Solution;
using sparse_net_library::Solution_solver;
using sparse_net_library::Index_synapse_interval;
using sparse_net_library::Input_synapse_interval;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Transfer_function;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::TRANSFER_FUNCTION_SIGMOID;
using sparse_net_library::TRANSFER_FUNCTION_TANH;
using sparse_net_library::TRANSFER_FUNCTION_RELU;
using sparse_net_library::TRANSFER_FUNCTION_SELU;
using sparse_net_library::NETWORK_RECURRENCE_TO_SELF;
using sparse_net_library::NETWORK_RECURRENCE_TO_LAYER;
using sparse_net_library::Spike_function;
using rafko_mainframe::Service_context;

using std::unique_ptr;
using std::make_unique;
using std::vector;

/*###############################################################################################
 * Testing if the solution solver produces a correct output, given a manually constructed
 * @Solution.
 * - 2 rows and two columns shall be constructed.
 * - @Partial_solution [0][0]: takes the whole of the input
 * - @Partial_solution [0][1]: takes half of the input
 * - @Partial_solution [1][0]: takes the whole of the previous row
 * - @Partial_solution [1][1]: takes half from each previous @Partial_solution
 */
void test_solution_solver_multithread(uint16 threads){
  Service_context service_context;

  /* Define the input, @Solution and partial solution table */
  Service_context context = Service_context().set_max_solve_threads(threads);
  Solution solution;
  solution.set_network_memory_length(1);
  solution.set_neuron_number(8);
  solution.set_output_neuron_number(4);
  solution.add_cols(2); /* Every row shall have 2 columns */
  solution.add_cols(2);
  *solution.add_partial_solutions() = Partial_solution();
  *solution.add_partial_solutions() = Partial_solution();
  *solution.add_partial_solutions() = Partial_solution();
  *solution.add_partial_solutions() = Partial_solution();

  vector<sdouble32> network_inputs = {double_literal(5.1),double_literal(10.3),double_literal(3.2),double_literal(9.4)};
  Input_synapse_interval temp_input_interval;

  /* [0][0]: Whole of the input */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(0), network_inputs.size(),0);
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0));
  temp_input_interval.set_interval_size(network_inputs.size());
  *solution.mutable_partial_solutions(0)->add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_0_0 = Partial_solution_solver(solution.partial_solutions(0), service_context);

  /* [0][1]: Half of the input */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(1), network_inputs.size()/2,2);
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(network_inputs.size()/2));
  temp_input_interval.set_interval_size(network_inputs.size()/2);
  *solution.mutable_partial_solutions(1)->add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_0_1 = Partial_solution_solver(solution.partial_solutions(1), service_context);

  /* [1][0]: Whole of the previous row's data --> neuron [0] to [3] */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(2),4,4);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(4);
  *solution.mutable_partial_solutions(2)->add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_1_0 = Partial_solution_solver(solution.partial_solutions(2), service_context);

  /* [1][1]: Half of the previous row's data ( in the middle) --> neuron [1] to [2] */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(3),2,6);
  temp_input_interval.set_starts(1);
  temp_input_interval.set_interval_size(2);
  *solution.mutable_partial_solutions(3)->add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_1_1 = Partial_solution_solver(solution.partial_solutions(3), service_context);

  /* Solve the compiled Solution */
  srand (time(nullptr));
  unique_ptr<Solution_solver> solution_solver(Solution_solver::Builder(solution, service_context).build());
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(solution.neuron_number());
  vector<sdouble32> network_output_vector;
  DataRingbuffer neuron_data_partials(1,8);
  DataRingbuffer neuron_data(1,8);

  for(uint8 variant_iterator = 0; variant_iterator < 100; variant_iterator++){
    if(0 < variant_iterator){ /* modify some weights biases and memory filters */
      for(sint32 i = 0; i < solution.partial_solutions(0).weight_table_size(); ++i){
        solution.mutable_partial_solutions(0)->set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(sint32 i = 0; i < solution.partial_solutions(1).weight_table_size(); ++i){
        solution.mutable_partial_solutions(1)->set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(sint32 i = 0; i < solution.partial_solutions(2).weight_table_size(); ++i){
        solution.mutable_partial_solutions(2)->set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(sint32 i = 0; i < solution.partial_solutions(3).weight_table_size(); ++i){
        solution.mutable_partial_solutions(3)->set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */

      /* Modify memory filters and transfer functions */
      solution.mutable_partial_solutions(0)->set_weight_table(solution.partial_solutions(0).memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(0)->set_weight_table(solution.partial_solutions(0).memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(0)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(0).neuron_transfer_functions_size()),Transfer_function::next());

      solution.mutable_partial_solutions(1)->set_weight_table(solution.partial_solutions(1).memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(1)->set_weight_table(solution.partial_solutions(1).memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(1)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(1).neuron_transfer_functions_size()),Transfer_function::next());

      solution.mutable_partial_solutions(2)->set_weight_table(solution.partial_solutions(2).memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(2)->set_weight_table(solution.partial_solutions(2).memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(2)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(2).neuron_transfer_functions_size()),Transfer_function::next());

      solution.mutable_partial_solutions(3)->set_weight_table(solution.partial_solutions(3).memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(3)->set_weight_table(solution.partial_solutions(3).memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      solution.mutable_partial_solutions(3)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(3).neuron_transfer_functions_size()),Transfer_function::next());
    }

    /* Calculate the expected output */
    manual_2_neuron_result( network_inputs,expected_neuron_data,solution.partial_solutions(0),0 ); /* row 0, column 0 */
    manual_2_neuron_result( {network_inputs.begin()+2,network_inputs.end()},expected_neuron_data,solution.partial_solutions(1),2 ); /* row 0, column 1 */
    manual_2_neuron_result( {expected_neuron_data.begin(),expected_neuron_data.begin() + 4},expected_neuron_data,solution.partial_solutions(2),4 ); /* row 1, column 0 */
    manual_2_neuron_result( {expected_neuron_data.begin() + 1,expected_neuron_data.begin() + 3},expected_neuron_data,solution.partial_solutions(3),6 ); /* row 1, column 1 */

    /* Solve the net */
    partial_solution_solver_0_0.solve(network_inputs, neuron_data_partials); /* row 0, column 0 */
    partial_solution_solver_0_1.solve(network_inputs, neuron_data_partials); /* row 0, column 1 */
    partial_solution_solver_1_0.solve(network_inputs, neuron_data_partials); /* row 1, column 0 */
    partial_solution_solver_1_1.solve(network_inputs, neuron_data_partials); /* row 1, column 1 */
    solution_solver->solve(network_inputs, neuron_data);

    /* Check result of the solution */
    REQUIRE( solution_solver->get_solution().output_neuron_number() <= neuron_data.get_element(0).size());
    network_output_vector = {
      neuron_data.get_const_element(0).end() - solution_solver->get_solution().output_neuron_number(),
      neuron_data.get_const_element(0).end()
    };
    REQUIRE( network_output_vector.size() == solution.output_neuron_number() );

    for(uint32 i = 0; i < network_output_vector.size(); ++i){
      REQUIRE(
        Approx(neuron_data_partials.get_element(0, solution.neuron_number() - solution.output_neuron_number() + i)).epsilon(double_literal(0.00000000000001))
        == expected_neuron_data[solution.neuron_number() - solution.output_neuron_number() + i]
      );
      REQUIRE(
        Approx(network_output_vector[i]).epsilon(double_literal(0.00000000000001))
        == expected_neuron_data[solution.neuron_number() - solution.output_neuron_number() + i]
      );
    }
  }
}

TEST_CASE("Solution solver manual testing","[solve][small][manual-solve]"){
  test_solution_solver_multithread(1);
  test_solution_solver_multithread(2);
  test_solution_solver_multithread(10);
}

/*###############################################################################################
 * Testing if the solution solver produces a correct output, given a built @SparseNet
 */
void testing_solution_solver_manually(google::protobuf::Arena* arena){
  Service_context service_context = Service_context()
  .set_max_solve_threads(4).set_device_max_megabytes(2048)
  .set_arena_ptr(arena);
  vector<uint32> net_structure = {20,40,30,10,20};
  vector<sdouble32> net_input = {double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)};

  /* Build the described net */
  SparseNet* net = Sparse_net_builder(service_context).input_size(5)
    .expected_input_range(double_literal(5.0)).dense_layers(net_structure);

  /* Generate solution from Net */
  Solution* solution = Solution_builder(service_context).build(*net);

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  unique_ptr<Solution_solver> solver(Solution_solver::Builder(*solution, service_context).build());
  DataRingbuffer neuron_data(1, solver->get_solution().neuron_number());
  DataRingbuffer neuron_data2(1, solver->get_solution().neuron_number());

  solver->solve(net_input, neuron_data);
  vector<sdouble32> result = {
    neuron_data.get_element(0).end() - solver->get_solution().output_neuron_number(),
    neuron_data.get_element(0).end()
  };
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(net->neuron_array_size());
  manaual_fully_connected_network_result(net_input, {}, expected_neuron_data, net_structure, *net);
  vector<sdouble32> expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};
  /* Verify if the calculated values match the expected ones */
  REQUIRE( net_structure.back() == result.size() );
  REQUIRE( expected_result.size() == result.size() );
  for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
    CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);

  /* Re-veriy with guaranteed multiple partial solutions */
  sdouble32 solution_size_mb = solution->SpaceUsedLong() /* Bytes */* double_literal(1024.0) /* KB */* double_literal(1024.0) /* MB */;
  (void)service_context.set_device_max_megabytes(solution_size_mb/double_literal(4.0));
  Solution* solution2 = Solution_builder(service_context).build(*net);

  unique_ptr<Solution_solver> solver2(Solution_solver::Builder(*solution2, service_context).build());
  solver2->solve(net_input, neuron_data2);
  result = {
    neuron_data2.get_element(0).end() - solver2->get_solution().output_neuron_number(),
    neuron_data2.get_element(0).end()
  };

  /* Verify once more if the calculated values match the expected ones */
  for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
    REQUIRE( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);

  if(nullptr == arena){
    delete solution2;
    delete solution;
    delete net;
  }
}

TEST_CASE("Solution Solver test based on Fully Connected Dense Net", "[solve][build-solve]"){
  testing_solution_solver_manually(nullptr);
}

/*###############################################################################################
 * Testing if the solution solver produces correct data for Networks generated
 * with connections of memories of the past
 *//* The utility function returns with the number of megabytes required for the complete Solution */
sdouble32 testing_nets_with_memory_manually(google::protobuf::Arena* arena, sdouble32 max_space_mb, uint32 recurrence){
  vector<uint32> net_structure = {20,30,40,30,20};
  vector<sdouble32> net_input = {
    double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)
  };

  /* Build the above described net */
  Service_context service_context = Service_context().set_device_max_megabytes(max_space_mb);
  Sparse_net_builder net_builder = Sparse_net_builder(service_context);
  net_builder.input_size(5).expected_input_range(double_literal(5.0));
  if(NETWORK_RECURRENCE_TO_SELF == recurrence)
    net_builder.set_recurrence_to_self();
  else if(NETWORK_RECURRENCE_TO_LAYER == recurrence)
    net_builder.set_recurrence_to_layer();

  SparseNet* net = net_builder.dense_layers(net_structure);

  /* Generate solution from Net */
  Solution* solution = Solution_builder(service_context).build(*net);
  unique_ptr<Solution_solver> solver(Solution_solver::Builder(*solution, service_context).build());
  DataRingbuffer neuron_data(solver->get_solution().network_memory_length(), solver->get_solution().neuron_number());

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  solver->solve(net_input, neuron_data);
  vector<sdouble32> result = {
    (neuron_data.get_element(0).end() - solver->get_solution().output_neuron_number()),
    neuron_data.get_element(0).end()
  };
  vector<sdouble32> previous_neuron_data = vector<sdouble32>(net->neuron_array_size());
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(net->neuron_array_size()); /* Should be all zeroes the first time */

  manaual_fully_connected_network_result(net_input, previous_neuron_data, expected_neuron_data, net_structure, *net);
  vector<sdouble32> expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};

  REQUIRE( net_structure.back() == result.size() );
  REQUIRE( expected_result.size() == result.size() );
  for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator){
    CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);
  }

  for(uint32 loop = 0; loop < 5; ++loop){ /* Re-verify with additional runs, at least 3, more shouldn't hurt */
    solver->solve(net_input, neuron_data);
    result = {neuron_data.get_element(0).end() - solver->get_solution().output_neuron_number(), neuron_data.get_element(0).end()};
    previous_neuron_data = vector<sdouble32>(expected_neuron_data);
    manaual_fully_connected_network_result(net_input, previous_neuron_data, expected_neuron_data, net_structure, *net);
    expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};

    REQUIRE( net_structure.back() == result.size() );
    REQUIRE( expected_result.size() == result.size() );
    for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
      CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);
  }

  /* Return with the size of the overall solution */
  sdouble32 space_used_mb = solution->SpaceUsedLong() /* Bytes */* double_literal(1024.0) /* KB */* double_literal(1024.0) /* MB */;

  if(nullptr == service_context.get_arena_ptr()){
    delete solution;
    delete net;
  }

  return space_used_mb;
}

TEST_CASE("Solution Solver test with memory", "[solve][memory]"){
  /* Test if the network is producing correct results when neurons take past-inputs from themselves ( 0x01 ID given to builder ) */
  sdouble32 megabytes_used = testing_nets_with_memory_manually(nullptr, (double_literal(4.0) * double_literal(1024.0)), 0x01);
  (void)testing_nets_with_memory_manually(nullptr, megabytes_used / double_literal(4.0),0x01); /* Even if the net needs to be splitted */

  /* Test if the network is producing correct results when neurons take past-inputs from their layers ( 0x02 ID given to builder ) */
  megabytes_used = testing_nets_with_memory_manually(nullptr, (double_literal(4.0) * double_literal(1024.0)), 0x02);
  (void)testing_nets_with_memory_manually(nullptr, megabytes_used / double_literal(4.0),0x02); /* Even if the net needs to be splitted */
}

/*###############################################################################################
 * Calculate a generated Fully Connected dense network manually by the network description
 * and compare the calculated results to the one provided by the solution.
 */
void test_generated_net_by_calculation(google::protobuf::Arena* arena){
  Service_context service_context = Service_context().set_arena_ptr(arena);
  vector<sdouble32> net_input = {
    double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)
  };
  vector<uint32> network_layout_sizes = {10,30,20};

  /* Generate a fully connected Neural network */
  unique_ptr<Sparse_net_builder> builder(make_unique<Sparse_net_builder>(service_context));
  builder->input_size(5)
    .output_neuron_number(20)
    .expected_input_range(double_literal(5.0));

  SparseNet* net(builder->dense_layers(
    network_layout_sizes,{
      {TRANSFER_FUNCTION_IDENTITY},
      {TRANSFER_FUNCTION_SELU,TRANSFER_FUNCTION_RELU},
      {TRANSFER_FUNCTION_TANH,TRANSFER_FUNCTION_SIGMOID}
    }
  ));

  /* Generate a solution */
  Solution* solution;
  REQUIRE_NOTHROW(
     solution = Solution_builder(service_context).build(*net)
  );
  service_context.set_device_max_megabytes( /* Introduce segmentation into the solution to test roboustness */
    (solution->SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB */ / double_literal(1024.0) /* MB */)/double_literal(4.0)
  );
  if(nullptr == arena){
    delete solution;
  }
  REQUIRE_NOTHROW(
     solution = Solution_builder(service_context).build(*net)
  );

  /* Solve the generated solution */
  unique_ptr<Solution_solver> solver(Solution_solver::Builder(*solution, service_context).build());
  DataRingbuffer network_output(1, solver->get_solution().neuron_number());

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  solver->solve(net_input, network_output);

  /* Calculate the network manually */
  Transfer_function transfer_function(service_context);
  const uint32 number_of_neurons = std::accumulate(network_layout_sizes.begin(),network_layout_sizes.end(),0);
  vector<sdouble32> manual_neuron_values = vector<sdouble32>(number_of_neurons);
  vector<bool> solved = vector<bool>(number_of_neurons, false);
  uint32 solved_neurons = 0u;
  uint32 solved_neurons_in_loop = -1;
  uint32 solved_inputs_in_neuron;
  uint32 overall_inputs_in_neuron;
  sint32 input_index;
  sdouble32 neuron_data;
  uint32 neuron_input_iterator = 0;
  while(
    (number_of_neurons > solved_neurons) /* Until all of the Neurons are solved */
    &&(0 < solved_neurons_in_loop) /* but in case no neurons could be solved in this loop, infinite loop is suspected */
  ){
    solved_neurons_in_loop = 0;
    /* Go for each neuron */
    for(uint32 neuron_iterator = 0; neuron_iterator < number_of_neurons; ++neuron_iterator){
      /* if the Neuron is solvable --> all of its children are etiher inputs or solved already */
      /* solve them, store its data and update the meta */
      if(false == solved[neuron_iterator]){
        Synapse_iterator<Input_synapse_interval> neuron_input_synapses(net->neuron_array(neuron_iterator).input_indices());
        overall_inputs_in_neuron = neuron_input_synapses.size();
        solved_inputs_in_neuron = 0;
        neuron_input_iterator = 0;
        neuron_data = 0;
        Synapse_iterator<>::iterate(net->neuron_array(neuron_iterator).input_weights(),
        [&](Index_synapse_interval weight_synapse, sint32 weight_index){
          if(neuron_input_iterator < neuron_input_synapses.size()){
            input_index = neuron_input_synapses[neuron_input_iterator];
            if(
              Synapse_iterator<>::is_index_input(input_index) /* Neuron input points to input data */
              ||(true == solved[input_index]) /* or the current input points to a neuron which is already solved */
            ){ /* the input counts as solved */
              ++solved_inputs_in_neuron;
            }
            if(Synapse_iterator<>::is_index_input(input_index)){
              input_index = Synapse_iterator<>::input_index_from_synapse_index(input_index);
              neuron_data += net_input[input_index] * net->weight_table(weight_index);
            }else{
              neuron_data += manual_neuron_values[input_index] * net->weight_table(weight_index);
            }
            ++neuron_input_iterator;
          }else{ /* After the inputs, every weight before the spike parameter is the bias */
            neuron_data += net->weight_table(weight_index);
          }
        });
        if(solved_inputs_in_neuron == overall_inputs_in_neuron){
          neuron_data = transfer_function.get_value(
            net->neuron_array(neuron_iterator).transfer_function_idx(),
            neuron_data
          );
          manual_neuron_values[neuron_iterator] = Spike_function::get_value(
            net->weight_table(net->neuron_array(neuron_iterator).memory_filter_idx()),
            neuron_data,
            manual_neuron_values[neuron_iterator]
          );
          solved[neuron_iterator] = true;
          ++solved_neurons;
          ++solved_neurons_in_loop;
        }
      } /*(false == solved[neuron_iterator])*//* if the condition is false, it means the neuron is already solved */
    }

  }/*while(the neurons are solved)*/
  REQUIRE(number_of_neurons == solved_neurons);

  /* Compare the calculated Neuron outputs to the values in the solution */
  for(uint32 neuron_index = 0; neuron_index < number_of_neurons; ++neuron_index){
    CHECK(manual_neuron_values[neuron_index] == network_output.get_element(0, neuron_index) );
  }

}

TEST_CASE("Solution Solver test with Generated fully connected network", "[solve][full]"){
  test_generated_net_by_calculation(nullptr);
}

}
