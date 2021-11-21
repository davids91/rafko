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

#include "rafko_protocol/solution.pb.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_service_context.h"
#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_net/models/transfer_function.h"
#include "rafko_net/models/spike_function.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_net/services/partial_solution_solver.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"

namespace rafko_net_test{

using rafko_mainframe::RafkoServiceContext;
using rafko_utilities::DataRingbuffer;
using rafko_utilities::ThreadGroup;
using rafko_net::RafkoNetBuilder;
using rafko_net::SolutionBuilder;
using rafko_net::RafkoNet;
using rafko_net::PartialSolution;
using rafko_net::PartialSolution_solver;
using rafko_net::Solution;
using rafko_net::SolutionSolver;
using rafko_net::IndexSynapseInterval;
using rafko_net::InputSynapseInterval;
using rafko_net::SynapseIterator;
using rafko_net::TransferFunction;
using rafko_net::transfer_function_identity;
using rafko_net::transfer_function_sigmoid;
using rafko_net::transfer_function_tanh;
using rafko_net::transfer_function_relu;
using rafko_net::transfer_function_selu;
using rafko_net::network_recurrence_to_self;
using rafko_net::network_recurrence_to_layer;
using rafko_net::SpikeFunction;

using std::unique_ptr;
using std::make_unique;
using std::vector;

/*###############################################################################################
 * Testing if the solution solver produces a correct output, given a manually constructed
 * @Solution.
 * - 2 rows and two columns shall be constructed.
 * - @PartialSolution [0][0]: takes the whole of the input
 * - @PartialSolution [0][1]: takes half of the input
 * - @PartialSolution [1][0]: takes the whole of the previous row
 * - @PartialSolution [1][1]: takes half from each previous @PartialSolution
 */
void test_solution_solver_multithread(uint16 threads){
  RafkoServiceContext service_context;

  /* Define the input, @Solution and partial solution table */
  RafkoServiceContext context = RafkoServiceContext().set_max_solve_threads(threads);
  Solution solution;
  solution.set_network_memory_length(1);
  solution.set_neuron_number(8);
  solution.set_output_neuron_number(4);
  solution.add_cols(2); /* Every row shall have 2 columns */
  solution.add_cols(2);
  *solution.add_partial_solutions() = PartialSolution();
  *solution.add_partial_solutions() = PartialSolution();
  *solution.add_partial_solutions() = PartialSolution();
  *solution.add_partial_solutions() = PartialSolution();

  vector<sdouble32> network_inputs = {double_literal(5.1),double_literal(10.3),double_literal(3.2),double_literal(9.4)};
  InputSynapseInterval temp_input_interval;

  /* [0][0]: Whole of the input */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(0), network_inputs.size(),0);
  temp_input_interval.set_starts(SynapseIterator<>::synapse_index_from_input_index(0));
  temp_input_interval.set_interval_size(network_inputs.size());
  *solution.mutable_partial_solutions(0)->add_input_data() = temp_input_interval;
  PartialSolution_solver partial_solution_solver_0_0 = PartialSolution_solver(solution.partial_solutions(0), service_context);

  /* [0][1]: Half of the input */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(1), network_inputs.size()/2,2);
  temp_input_interval.set_starts(SynapseIterator<>::synapse_index_from_input_index(network_inputs.size()/2));
  temp_input_interval.set_interval_size(network_inputs.size()/2);
  *solution.mutable_partial_solutions(1)->add_input_data() = temp_input_interval;
  PartialSolution_solver partial_solution_solver_0_1 = PartialSolution_solver(solution.partial_solutions(1), service_context);

  /* [1][0]: Whole of the previous row's data --> neuron [0] to [3] */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(2),4,4);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(4);
  *solution.mutable_partial_solutions(2)->add_input_data() = temp_input_interval;
  PartialSolution_solver partial_solution_solver_1_0 = PartialSolution_solver(solution.partial_solutions(2), service_context);

  /* [1][1]: Half of the previous row's data ( in the middle) --> neuron [1] to [2] */
  manual_2_neuron_partial_solution(*solution.mutable_partial_solutions(3),2,6);
  temp_input_interval.set_starts(1);
  temp_input_interval.set_interval_size(2);
  *solution.mutable_partial_solutions(3)->add_input_data() = temp_input_interval;
  PartialSolution_solver partial_solution_solver_1_1 = PartialSolution_solver(solution.partial_solutions(3), service_context);

  /* Solve the compiled Solution */
  srand (time(nullptr));
  unique_ptr<SolutionSolver> solution_solver(SolutionSolver::Builder(solution, service_context).build());
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(solution.neuron_number());
  vector<sdouble32> network_output_vector;
  DataRingbuffer neuron_data_partials(1,8);

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

      /* Modify transfer functions */
      solution.mutable_partial_solutions(0)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(0).neuron_transfer_functions_size()),TransferFunction::next());
      solution.mutable_partial_solutions(1)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(1).neuron_transfer_functions_size()),TransferFunction::next());
      solution.mutable_partial_solutions(2)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(2).neuron_transfer_functions_size()),TransferFunction::next());
      solution.mutable_partial_solutions(3)->set_neuron_transfer_functions(rand()%(solution.partial_solutions(3).neuron_transfer_functions_size()),TransferFunction::next());
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
    rafko_utilities::ConstVectorSubrange<sdouble32> neuron_data = solution_solver->solve(network_inputs, false);

    /* Check result of the solution */
    REQUIRE( solution_solver->get_solution().output_neuron_number() <= neuron_data.size());
    network_output_vector = {
      neuron_data.cend() - solution_solver->get_solution().output_neuron_number(),
      neuron_data.cend()
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
 * Testing if the solution solver produces a correct output, given a built @RafkoNet
 */
void testing_solution_solver_manually(google::protobuf::Arena* arena){
  RafkoServiceContext service_context = RafkoServiceContext()
  .set_max_solve_threads(4).set_device_max_megabytes(2048)
  .set_arena_ptr(arena);
  vector<uint32> net_structure = {20,40,30,10,20};
  vector<sdouble32> net_input = {double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)};

  /* Build the described net */
  RafkoNet* net = RafkoNetBuilder(service_context).input_size(5)
    .expected_input_range(double_literal(5.0)).dense_layers(net_structure);

  /* Generate solution from Net */
  Solution* solution = SolutionBuilder(service_context).build(*net);

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  unique_ptr<SolutionSolver> solver(SolutionSolver::Builder(*solution, service_context).build());

  rafko_utilities::ConstVectorSubrange<sdouble32> neuron_data = solver->solve(net_input, true);
  vector<sdouble32> result = {
    neuron_data.cend() - solver->get_solution().output_neuron_number(),
    neuron_data.cend()
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
  Solution* solution2 = SolutionBuilder(service_context).build(*net);

  unique_ptr<SolutionSolver> solver2(SolutionSolver::Builder(*solution2, service_context).build());
  rafko_utilities::ConstVectorSubrange<sdouble32> neuron_data2 = solver2->solve(net_input, true);
  result = {
    neuron_data2.cend() - solver2->get_solution().output_neuron_number(),
    neuron_data2.cend()
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
  RafkoServiceContext service_context = RafkoServiceContext().set_arena_ptr(arena).set_device_max_megabytes(max_space_mb);
  RafkoNetBuilder net_builder = RafkoNetBuilder(service_context);
  net_builder.input_size(5).expected_input_range(double_literal(5.0));
  if(network_recurrence_to_self == recurrence)
    net_builder.set_recurrence_to_self();
  else if(network_recurrence_to_layer == recurrence)
    net_builder.set_recurrence_to_layer();

  RafkoNet* net = net_builder.dense_layers(net_structure);

  /* Generate solution from Net */
  Solution* solution = SolutionBuilder(service_context).build(*net);
  unique_ptr<SolutionSolver> solver(SolutionSolver::Builder(*solution, service_context).build());

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  rafko_utilities::ConstVectorSubrange<sdouble32> neuron_data = solver->solve(net_input, true);
  vector<sdouble32> result = {
    (neuron_data.cend() - solver->get_solution().output_neuron_number()),
    neuron_data.cend()
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
    rafko_utilities::ConstVectorSubrange<sdouble32> neuron_data = solver->solve(net_input, false);
    result = {neuron_data.cend() - solver->get_solution().output_neuron_number(), neuron_data.cend()};
    previous_neuron_data = vector<sdouble32>(expected_neuron_data);
    manaual_fully_connected_network_result(net_input, previous_neuron_data, expected_neuron_data, net_structure, *net);
    expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};

    REQUIRE( net_structure.back() == result.size() );
    REQUIRE( expected_result.size() == result.size() );
    for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
      REQUIRE( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);
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
  RafkoServiceContext service_context = RafkoServiceContext().set_arena_ptr(arena);
  vector<sdouble32> net_input = {
    double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)
  };
  vector<uint32> network_layout_sizes = {10,30,20};

  /* Generate a fully connected Neural network */
  unique_ptr<RafkoNetBuilder> builder(make_unique<RafkoNetBuilder>(service_context));
  builder->input_size(5)
    .output_neuron_number(network_layout_sizes.back())
    .expected_input_range(double_literal(5.0));

  RafkoNet* net(builder->dense_layers(
    network_layout_sizes,{
      {transfer_function_identity},
      {transfer_function_selu,transfer_function_relu},
      {transfer_function_tanh,transfer_function_sigmoid}
    }
  ));

  /* Generate a solution */
  Solution* solution;
  REQUIRE_NOTHROW(
     solution = SolutionBuilder(service_context).build(*net)
  );
  service_context.set_device_max_megabytes( /* Introduce segmentation into the solution to test roboustness */
    (solution->SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB */ / double_literal(1024.0) /* MB */)/double_literal(4.0)
  );
  if(nullptr == arena){
    delete solution;
  }
  REQUIRE_NOTHROW(
     solution = SolutionBuilder(service_context).build(*net)
  );

  /* Solve the generated solution */
  unique_ptr<SolutionSolver> solver(SolutionSolver::Builder(*solution, service_context).build());

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  rafko_utilities::ConstVectorSubrange<sdouble32> network_output = solver->solve(net_input, true);

  /* Calculate the network manually */
  TransferFunction transfer_function(service_context);
  const uint32 number_of_neurons = std::accumulate(network_layout_sizes.begin(),network_layout_sizes.end(),0);
  vector<sdouble32> manual_neuron_values = vector<sdouble32>(number_of_neurons);
  vector<bool> solved = vector<bool>(number_of_neurons, false);
  uint32 solved_neurons = 0u;
  uint32 solved_neurons_in_loop = -1;
  uint32 solved_inputs_in_neuron;
  uint32 overall_inputs_in_neuron;
  sint32 input_index;
  sdouble32 neuron_data;
  sdouble32 spike_function_weight;
  uint32 neuron_input_iterator = 0;
  bool first_weight_in_synapse;
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
        SynapseIterator<InputSynapseInterval> neuron_input_synapses(net->neuron_array(neuron_iterator).input_indices());
        overall_inputs_in_neuron = neuron_input_synapses.size();
        solved_inputs_in_neuron = 0;
        neuron_input_iterator = 0;
        neuron_data = 0;
        first_weight_in_synapse = true;
        spike_function_weight = double_literal(0.0);
        SynapseIterator<>::iterate(net->neuron_array(neuron_iterator).input_weights(),
        [&](sint32 weight_index){
          if(true == first_weight_in_synapse){
            first_weight_in_synapse = false;
            spike_function_weight = net->weight_table(weight_index);
          }else{
            if(neuron_input_iterator < neuron_input_synapses.size()){
              input_index = neuron_input_synapses[neuron_input_iterator];
              if(
                SynapseIterator<>::is_index_input(input_index) /* Neuron input points to input data */
                ||(true == solved[input_index]) /* or the current input points to a neuron which is already solved */
              ){ /* the input counts as solved */
                ++solved_inputs_in_neuron;
              }
              if(SynapseIterator<>::is_index_input(input_index)){
                input_index = SynapseIterator<>::input_index_from_synapse_index(input_index);
                neuron_data += net_input[input_index] * net->weight_table(weight_index);
              }else{
                neuron_data += manual_neuron_values[input_index] * net->weight_table(weight_index);
              }
              ++neuron_input_iterator;
            }else{ /* After the inputs, every weight is the bias */
              neuron_data += net->weight_table(weight_index);
            }
          }
        });
        if(solved_inputs_in_neuron == overall_inputs_in_neuron){
          neuron_data = transfer_function.get_value(
            net->neuron_array(neuron_iterator).transfer_function_idx(),
            neuron_data
          );
          manual_neuron_values[neuron_iterator] = SpikeFunction::get_value(
            spike_function_weight, neuron_data, manual_neuron_values[neuron_iterator]
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
  for(uint32 neuron_index = 0; neuron_index < network_layout_sizes.back(); ++neuron_index){
    REQUIRE( /* Solution solver only provides the data of the output neurons! */
      manual_neuron_values[number_of_neurons - network_layout_sizes.back() + neuron_index]
      == network_output[neuron_index]
    );
  }

}

TEST_CASE("Solution Solver test with Generated fully connected network", "[solve][full]"){
  test_generated_net_by_calculation(nullptr);
}

/*###############################################################################################
 * Test if the solver is able to produce correct output when used from multiple threads
 */
TEST_CASE("Solution Solver Multi-threading test", "[solve][full][multithread]"){
  vector<uint32> net_structure = {20,30,40,30,20};
  vector<sdouble32> net_input = {
    double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)
  };
  RafkoServiceContext service_context = RafkoServiceContext();
  RafkoNet* net = RafkoNetBuilder(service_context)
    .input_size(5).expected_input_range(double_literal(5.0))
    .dense_layers(net_structure);
  Solution* solution = SolutionBuilder(service_context).build(*net);
  unique_ptr<SolutionSolver> solver(SolutionSolver::Builder(*solution, service_context).build());

  /* solve in a single thread */
  rafko_utilities::ConstVectorSubrange<sdouble32> single_thread_output_buffer = solver->solve(net_input, true);
  const vector<sdouble32> single_thread_output = {
    single_thread_output_buffer.cbegin(),
    single_thread_output_buffer.cend()
  };

  /* solve from multiple threads */
  const uint32 thread_number = service_context.get_max_processing_threads();
  ThreadGroup executor(thread_number);
  vector<vector<sdouble32>> thread_outputs(thread_number);
  executor.start_and_block([&](uint32 thread_index){
    rafko_utilities::ConstVectorSubrange<sdouble32> thread_output_buffer = solver->solve(net_input, true, thread_index);
    thread_outputs[thread_index] = {
      thread_output_buffer.cbegin(),
      thread_output_buffer.cend()
    };
  });

  /* compare that multi-thread solve should be the same as single thread solve */
  for(uint32 neuron_data_index = 0; neuron_data_index < single_thread_output.size(); ++neuron_data_index){
    for(uint32 thread_index = 0; thread_index < thread_number; ++thread_index){
      REQUIRE( single_thread_output[neuron_data_index] == thread_outputs[thread_index][neuron_data_index] );
    }
  }
}

/*###############################################################################################
 * Test if the solver is able to remember the previous neuron values correctly
 */
TEST_CASE("Solution Solver memory test", "[solve][memory]"){
  RafkoServiceContext service_context = RafkoServiceContext();
  RafkoNet* net = RafkoNetBuilder(service_context)
    .input_size(1).expected_input_range(double_literal(5.0))
    .set_recurrence_to_self()
    .allowed_transfer_functions_by_layer({{transfer_function_identity}})
    .dense_layers({1});

  for(sint32 weight_index = 0; weight_index < net->weight_table_size(); ++weight_index){
    net->set_weight_table(weight_index, double_literal(1.0));
  }
  net->set_weight_table(0u,double_literal(0.0)); /* Set the memory filter of the only neuron to 0, so the previous value of it would not modify the current one through the spike function */

  Solution* solution = SolutionBuilder(service_context).build(*net);
  unique_ptr<SolutionSolver> solver(SolutionSolver::Builder(*solution, service_context).build());

  sdouble32 expected_result = double_literal(1.0);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    CHECK( expected_result ==  (solver->solve({double_literal(0.0)}, false, 0u))[0]);
    expected_result += double_literal(1.0);
  }
}

}
