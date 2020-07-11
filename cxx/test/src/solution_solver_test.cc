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

#include <vector>
#include <memory>

#include "test/test_utility.h"

#include "gen/solution.pb.h"
#include "gen/sparse_net.pb.h"
#include "sparse_net_library/models/transfer_function.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/data_ringbuffer.h"
#include "sparse_net_library/services/solution_solver.h"
#include "sparse_net_library/services/partial_solution_solver.h"
#include "sparse_net_library/services/synapse_iterator.h"
#include "sparse_net_library/services/sparse_net_builder.h"
#include "sparse_net_library/services/solution_builder.h"

namespace sparse_net_library_test{

using std::reference_wrapper;
using std::unique_ptr;
using std::make_unique;

using sparse_net_library::Sparse_net_builder;
using sparse_net_library::Solution_builder;
using sparse_net_library::SparseNet;
using sparse_net_library::Data_ringbuffer;
using sparse_net_library::Partial_solution;
using sparse_net_library::Partial_solution_solver;
using sparse_net_library::Solution;
using sparse_net_library::Solution_solver;
using sparse_net_library::Index_synapse_interval;
using sparse_net_library::Input_synapse_interval;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Transfer_function;
using sparse_net_library::COST_FUNCTION_MSE;
using sparse_net_library::NETWORK_RECURRENCE_TO_SELF;
using sparse_net_library::NETWORK_RECURRENCE_TO_LAYER;
using rafko_mainframe::Service_context;

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

  /* Define the input, @Solution and partial solution table */
  Data_ringbuffer neuron_data(1,8);
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

  vector<vector<reference_wrapper<Partial_solution>>> partial_solutions= {
    {*solution.mutable_partial_solutions(0),*solution.mutable_partial_solutions(1)},
    {*solution.mutable_partial_solutions(2),*solution.mutable_partial_solutions(3)}
  };

  vector<sdouble32> network_inputs = {double_literal(5.1),double_literal(10.3),double_literal(3.2),double_literal(9.4)};
  Input_synapse_interval temp_input_interval;

  /* [0][0]: Whole of the input */
  manual_2_neuron_partial_solution(partial_solutions[0][0], network_inputs.size(),0);
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0));
  temp_input_interval.set_interval_size(network_inputs.size());
  *partial_solutions[0][0].get().add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_0_0 = Partial_solution_solver(partial_solutions[0][0], neuron_data);

  /* [0][1]: Half of the input */
  manual_2_neuron_partial_solution(partial_solutions[0][1], network_inputs.size()/2,2);
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(network_inputs.size()/2));
  temp_input_interval.set_interval_size(network_inputs.size()/2);
  *partial_solutions[0][1].get().add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_0_1 = Partial_solution_solver(partial_solutions[0][1], neuron_data);

  /* [1][0]: Whole of the previous row's data --> neuron [0] to [3] */
  manual_2_neuron_partial_solution(partial_solutions[1][0],4,4);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(4);
  *partial_solutions[1][0].get().add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_1_0 = Partial_solution_solver(partial_solutions[1][0], neuron_data);

  /* [1][1]: Half of the previous row's data ( in the middle) --> neuron [1] to [2] */
  manual_2_neuron_partial_solution(partial_solutions[1][1],2,6);
  temp_input_interval.set_starts(1);
  temp_input_interval.set_interval_size(2);
  *partial_solutions[1][1].get().add_input_data() = temp_input_interval;
  Partial_solution_solver partial_solution_solver_1_1 = Partial_solution_solver(partial_solutions[1][1], neuron_data);

  /* Solve the compiled Solution */
  srand (time(nullptr));
  Solution_solver solution_solver(solution,context);
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(solution.neuron_number());
  vector<sdouble32> network_output;

  for(uint8 variant_iterator = 0; variant_iterator < 100; variant_iterator++){
    if(0 < variant_iterator){ /* modify some weights biases and memory filters */
      for(int i = 0; i < partial_solutions[0][0].get().weight_table_size(); ++i){
        partial_solutions[0][0].get().set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(int i = 0; i < partial_solutions[0][1].get().weight_table_size(); ++i){
        partial_solutions[0][1].get().set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(int i = 0; i < partial_solutions[1][0].get().weight_table_size(); ++i){
        partial_solutions[1][0].get().set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */
      for(int i = 0; i < partial_solutions[1][1].get().weight_table_size(); ++i){
        partial_solutions[1][1].get().set_weight_table(i,static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      } /* Modify weights */

      /* Modify memory filters and transfer functions */
      partial_solutions[0][0].get().set_weight_table(partial_solutions[0][0].get().memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[0][0].get().set_weight_table(partial_solutions[0][0].get().memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[0][0].get().set_neuron_transfer_functions(rand()%(partial_solutions[0][0].get().neuron_transfer_functions_size()),Transfer_function::next());

      partial_solutions[0][1].get().set_weight_table(partial_solutions[0][1].get().memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[0][1].get().set_weight_table(partial_solutions[0][1].get().memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[0][1].get().set_neuron_transfer_functions(rand()%(partial_solutions[0][1].get().neuron_transfer_functions_size()),Transfer_function::next());

      partial_solutions[1][0].get().set_weight_table(partial_solutions[1][0].get().memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[1][0].get().set_weight_table(partial_solutions[1][0].get().memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[1][0].get().set_neuron_transfer_functions(rand()%(partial_solutions[1][0].get().neuron_transfer_functions_size()),Transfer_function::next());

      partial_solutions[1][1].get().set_weight_table(partial_solutions[1][1].get().memory_filter_index(0),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[1][1].get().set_weight_table(partial_solutions[1][1].get().memory_filter_index(1),static_cast<sdouble32>(rand()%11) / double_literal(10.0));
      partial_solutions[1][1].get().set_neuron_transfer_functions(rand()%(partial_solutions[1][1].get().neuron_transfer_functions_size()),Transfer_function::next());
    }

    /* Calculate the expected output */
    manual_2_neuron_result(
      network_inputs,expected_neuron_data,partial_solutions[0][0],0
    ); /* row 0, column 0 */
    manual_2_neuron_result(
      {network_inputs.begin()+2,network_inputs.end()},expected_neuron_data,partial_solutions[0][1],2
    ); /* row 0, column 1 */
    manual_2_neuron_result(
      {expected_neuron_data.begin(),expected_neuron_data.begin() + 4},expected_neuron_data,partial_solutions[1][0],4
    ); /* row 1, column 0 */
    manual_2_neuron_result(
      {expected_neuron_data.begin() + 1,expected_neuron_data.begin() + 3},expected_neuron_data,partial_solutions[1][1],6
    ); /* row 1, column 1 */

    /* Solve the net */
    /* row 0, column 0 */
    partial_solution_solver_0_0.collect_input_data(network_inputs);
    partial_solution_solver_0_0.solve();
    partial_solution_solver_0_0.provide_output_data();

    /* row 0, column 1 */
    partial_solution_solver_0_1.collect_input_data(network_inputs);
    partial_solution_solver_0_1.solve();
    partial_solution_solver_0_1.provide_output_data();

    /* row 1, column 0 */
    partial_solution_solver_1_0.collect_input_data(network_inputs);
    partial_solution_solver_1_0.solve();
    partial_solution_solver_1_0.provide_output_data();

    /* row 1, column 1 */
    partial_solution_solver_1_1.collect_input_data(network_inputs);
    partial_solution_solver_1_1.solve();
    partial_solution_solver_1_1.provide_output_data();

    solution_solver.solve(network_inputs);

    network_output = solution_solver.get_neuron_data();
    REQUIRE( solution_solver.get_output_size() <= solution_solver.get_neuron_data().size());

    network_output = {network_output.end() - solution_solver.get_output_size(),network_output.end()};
    REQUIRE( network_output.size() == solution.output_neuron_number() );

    for(uint32 i = 0; i < network_output.size(); ++i){
      CHECK(
        Approx(neuron_data.get_element(solution.neuron_number() - solution.output_neuron_number() + i, 0)).epsilon(double_literal(0.00000000000001))
        == expected_neuron_data[solution.neuron_number() - solution.output_neuron_number() + i]
      );
      CHECK(
        Approx(network_output[i]).epsilon(double_literal(0.00000000000001))
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
  vector<uint32> net_structure = {2,4,3,1,2};
  vector<sdouble32> net_input = {double_literal(10.0),double_literal(20.0),double_literal(30.0),double_literal(40.0),double_literal(50.0)};

  /* Build the described net */
  Sparse_net_builder net_builder = Sparse_net_builder();
  net_builder.input_size(5).expected_input_range(double_literal(5.0))
  .cost_function(COST_FUNCTION_MSE).arena_ptr(arena);
  unique_ptr<SparseNet> net(net_builder.dense_layers(net_structure));

  /* Generate solution from Net */
  unique_ptr<Solution_builder> solution_builder = make_unique<Solution_builder>();
  Solution solution = *solution_builder->max_solve_threads(4).device_max_megabytes(2048).arena_ptr(arena).build(*net);

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  Solution_solver solver(solution);
  solver.solve(net_input);
  vector<sdouble32> result = {solver.get_neuron_data().end() - solver.get_output_size(), solver.get_neuron_data().end()};
  vector<sdouble32> expected_neuron_data = vector<sdouble32>(net->neuron_array_size());
  manaual_fully_connected_network_result(net_input, {}, expected_neuron_data, net_structure, *net);
  vector<sdouble32> expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};
  /* Verify if the calculated values match the expected ones */
  REQUIRE( net_structure.back() == result.size() );
  REQUIRE( expected_result.size() == result.size() );
  for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
    CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);

  /* Re-veriy with guaranteed multiple partial solutions */
  sdouble32 solution_size = solution.SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */;
  Solution solution2 = *solution_builder->max_solve_threads(4).device_max_megabytes(solution_size/double_literal(4.0)).arena_ptr(arena).build(*net);
  Solution_solver solver2(solution2);
  solver2.solve(net_input);
  result = {solver2.get_neuron_data().end() - solver2.get_output_size(),solver2.get_neuron_data().end()};
  
  /* Verify once more if the calculated values match the expected ones */
  for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
    CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);
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
  Sparse_net_builder net_builder = Sparse_net_builder();
  net_builder.input_size(5).expected_input_range(double_literal(5.0)).cost_function(COST_FUNCTION_MSE);
  if(NETWORK_RECURRENCE_TO_SELF == recurrence)
    net_builder.set_recurrence_to_self();
  else if(NETWORK_RECURRENCE_TO_LAYER == recurrence)
    net_builder.set_recurrence_to_layer();

  unique_ptr<SparseNet> net(net_builder.dense_layers(net_structure));

  /* Generate solution from Net */
  unique_ptr<Solution> solution = unique_ptr<Solution>(
    Solution_builder().service_context().device_max_megabytes(max_space_mb).build(*net)
  );
  Solution_solver solver(*solution);

  /* Verify if a generated solution gives back the exact same result, as the manually calculated one */
  solver.solve(net_input);
  REQUIRE( net->neuron_array_size() == static_cast<sint32>(solver.get_transfer_function_input().size()) );
  REQUIRE( net->neuron_array_size() == static_cast<sint32>(solver.get_transfer_function_output().size()) );
  vector<sdouble32> result = {solver.get_neuron_data().end() - solver.get_output_size(), solver.get_neuron_data().end()};
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
    solver.solve(net_input);
    result = {solver.get_neuron_data().end() - solver.get_output_size(), solver.get_neuron_data().end()};
    previous_neuron_data = vector<sdouble32>(expected_neuron_data);
    manaual_fully_connected_network_result(net_input, previous_neuron_data, expected_neuron_data, net_structure, *net);
    expected_result = {expected_neuron_data.end() - net->output_neuron_number(), expected_neuron_data.end()};

    REQUIRE( net_structure.back() == result.size() );
    REQUIRE( expected_result.size() == result.size() );
    for(uint32 result_iterator = 0; result_iterator < expected_result.size(); ++result_iterator)
      CHECK( Approx(result[result_iterator]).epsilon(double_literal(0.00000000000001)) == expected_result[result_iterator]);
  }

  /* Return with the size of the overall solution */
  return solution->SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */;
}

TEST_CASE("Solution Solver test with memory", "[solve][memory]"){
  /* Test if the network is producing correct results when neurons take past-inputs from themselves ( 0x01 ID given to builder ) */
  sdouble32 megabytes_used = testing_nets_with_memory_manually(nullptr, (double_literal(4.0) * double_literal(1024.0)), 0x01);
  (void)testing_nets_with_memory_manually(nullptr, megabytes_used / double_literal(4.0),0x01); /* Even if the net needs to be splitted */

  // /* Test if the network is producing correct results when neurons take past-inputs from their layers ( 0x02 ID given to builder ) */
  // megabytes_used = testing_nets_with_memory_manually(nullptr, (double_literal(4.0) * double_literal(1024.0)), 0x02);
  // (void)testing_nets_with_memory_manually(nullptr, megabytes_used / double_literal(4.0),0x02); /* Even if the net needs to be splitted */
}

}
