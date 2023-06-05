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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <numeric>
#include <vector>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/models/spike_function.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_net/services/partial_solution_solver.hpp"
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_net/services/solution_solver.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#include "rafko_utilities/services/thread_group.hpp"

#include "test/test_utility.hpp"

namespace rafko_net_test {

/*###############################################################################################
 * Testing if the solution solver produces a correct output, given a manually
 * constructed
 * @Solution.
 * - 2 rows and two columns shall be constructed.
 * - @PartialSolution [0][0]: takes the whole of the input
 * - @PartialSolution [0][1]: takes half of the input
 * - @PartialSolution [1][0]: takes the whole of the previous row
 * - @PartialSolution [1][1]: takes half from each previous @PartialSolution
 */
void test_solution_solver_multithread(std::uint16_t threads) {
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_max_solve_threads(threads);

  /* Define the input, @Solution and partial solution table */
  rafko_net::Solution solution;
  solution.set_network_memory_length(1);
  solution.set_neuron_number(8);
  solution.set_output_neuron_number(4);
  solution.set_network_input_size(4);
  solution.add_cols(2); /* Every row shall have 2 columns */
  solution.add_cols(2);
  *solution.add_partial_solutions() = rafko_net::PartialSolution();
  *solution.add_partial_solutions() = rafko_net::PartialSolution();
  *solution.add_partial_solutions() = rafko_net::PartialSolution();
  *solution.add_partial_solutions() = rafko_net::PartialSolution();

  std::vector<double> network_inputs = {(5.1), (10.3), (3.2), (9.4)};
  rafko_net::InputSynapseInterval temp_input_interval;

  /* [0][0]: Whole of the input */
  rafko_test::manual_2_neuron_partial_solution(
      *solution.mutable_partial_solutions(0), network_inputs.size(), 0);
  temp_input_interval.set_starts(
      rafko_net::SynapseIterator<>::external_index_from_array_index(0));
  temp_input_interval.set_interval_size(network_inputs.size());
  *solution.mutable_partial_solutions(0)->add_input_data() =
      temp_input_interval;
  rafko_net::PartialSolutionSolver partial_solution_solver_0_0 =
      rafko_net::PartialSolutionSolver(solution.partial_solutions(0), settings);

  /* [0][1]: Half of the input */
  rafko_test::manual_2_neuron_partial_solution(
      *solution.mutable_partial_solutions(1), network_inputs.size() / 2, 2);
  temp_input_interval.set_starts(
      rafko_net::SynapseIterator<>::external_index_from_array_index(
          network_inputs.size() / 2));
  temp_input_interval.set_interval_size(network_inputs.size() / 2);
  *solution.mutable_partial_solutions(1)->add_input_data() =
      temp_input_interval;
  rafko_net::PartialSolutionSolver partial_solution_solver_0_1 =
      rafko_net::PartialSolutionSolver(solution.partial_solutions(1), settings);

  /* [1][0]: Whole of the previous row's data --> neuron [0] to [3] */
  rafko_test::manual_2_neuron_partial_solution(
      *solution.mutable_partial_solutions(2), 4, 4);
  temp_input_interval.set_starts(0);
  temp_input_interval.set_interval_size(4);
  *solution.mutable_partial_solutions(2)->add_input_data() =
      temp_input_interval;
  rafko_net::PartialSolutionSolver partial_solution_solver_1_0 =
      rafko_net::PartialSolutionSolver(solution.partial_solutions(2), settings);

  /* [1][1]: Half of the previous row's data ( in the middle) --> neuron [1] to
   * [2] */
  rafko_test::manual_2_neuron_partial_solution(
      *solution.mutable_partial_solutions(3), 2, 6);
  temp_input_interval.set_starts(1);
  temp_input_interval.set_interval_size(2);
  *solution.mutable_partial_solutions(3)->add_input_data() =
      temp_input_interval;
  rafko_net::PartialSolutionSolver partial_solution_solver_1_1 =
      rafko_net::PartialSolutionSolver(solution.partial_solutions(3), settings);

  /* Solve the compiled Solution */
  srand(time(nullptr));
  std::shared_ptr<rafko_net::SolutionSolver> solution_solver =
      std::make_unique<rafko_net::SolutionSolver>(&solution, settings);
  std::vector<double> expected_neuron_data =
      std::vector<double>(solution.neuron_number());
  std::vector<double> network_output_vector;
  rafko_utilities::DataRingbuffer<> neuron_data_partials(
      1u, [](std::vector<double> &element) { element.resize(8u); });

  for (std::uint8_t variant_iterator = 0; variant_iterator < 10u;
       variant_iterator++) {
    if (0 <
        variant_iterator) { /* modify some weights biases and memory filters */
      for (std::int32_t i = 0;
           i < solution.partial_solutions(0).weight_table_size(); ++i) {
        solution.mutable_partial_solutions(0)->set_weight_table(
            i, static_cast<double>(rand() % 11) / 10.0);
      } /* Modify weights */
      for (std::int32_t i = 0;
           i < solution.partial_solutions(1).weight_table_size(); ++i) {
        solution.mutable_partial_solutions(1)->set_weight_table(
            i, static_cast<double>(rand() % 11) / 10.0);
      } /* Modify weights */
      for (std::int32_t i = 0;
           i < solution.partial_solutions(2).weight_table_size(); ++i) {
        solution.mutable_partial_solutions(2)->set_weight_table(
            i, static_cast<double>(rand() % 11) / 10.0);
      } /* Modify weights */
      for (std::int32_t i = 0;
           i < solution.partial_solutions(3).weight_table_size(); ++i) {
        solution.mutable_partial_solutions(3)->set_weight_table(
            i, static_cast<double>(rand() % 11) / 10.0);
      } /* Modify weights */

      /* Modify transfer functions */
      solution.mutable_partial_solutions(0)->set_neuron_transfer_functions(
          rand() %
              (solution.partial_solutions(0).neuron_transfer_functions_size()),
          rafko_net::TransferFunction::next());
      solution.mutable_partial_solutions(1)->set_neuron_transfer_functions(
          rand() %
              (solution.partial_solutions(1).neuron_transfer_functions_size()),
          rafko_net::TransferFunction::next());
      solution.mutable_partial_solutions(2)->set_neuron_transfer_functions(
          rand() %
              (solution.partial_solutions(2).neuron_transfer_functions_size()),
          rafko_net::TransferFunction::next());
      solution.mutable_partial_solutions(3)->set_neuron_transfer_functions(
          rand() %
              (solution.partial_solutions(3).neuron_transfer_functions_size()),
          rafko_net::TransferFunction::next());
    }

    /* Calculate the expected output */
    rafko_test::manual_2_neuron_result(network_inputs, expected_neuron_data,
                                       solution.partial_solutions(0),
                                       0); /* row 0, column 0 */
    rafko_test::manual_2_neuron_result(
        {network_inputs.begin() + 2, network_inputs.end()},
        expected_neuron_data, solution.partial_solutions(1),
        2); /* row 0, column 1 */
    rafko_test::manual_2_neuron_result(
        {expected_neuron_data.begin(), expected_neuron_data.begin() + 4},
        expected_neuron_data, solution.partial_solutions(2),
        4); /* row 1, column 0 */
    rafko_test::manual_2_neuron_result(
        {expected_neuron_data.begin() + 1, expected_neuron_data.begin() + 3},
        expected_neuron_data, solution.partial_solutions(3),
        6); /* row 1, column 1 */

    /* Solve the net */
    partial_solution_solver_0_0.solve(
        network_inputs, neuron_data_partials); /* row 0, column 0 */
    partial_solution_solver_0_1.solve(
        network_inputs, neuron_data_partials); /* row 0, column 1 */
    partial_solution_solver_1_0.solve(
        network_inputs, neuron_data_partials); /* row 1, column 0 */
    partial_solution_solver_1_1.solve(
        network_inputs, neuron_data_partials); /* row 1, column 1 */
    rafko_utilities::ConstVectorSubrange<> neuron_data =
        solution_solver->solve(network_inputs, false);

    /* Check result of the solution */
    REQUIRE(solution.output_neuron_number() <= neuron_data.size());
    network_output_vector = {
        neuron_data.end() - solution.output_neuron_number(), neuron_data.end()};
    REQUIRE(network_output_vector.size() == solution.output_neuron_number());

    for (std::uint32_t i = 0; i < network_output_vector.size(); ++i) {
      REQUIRE(Catch::Approx(neuron_data_partials.get_element(
                                0, solution.neuron_number() -
                                       solution.output_neuron_number() + i))
                  .epsilon((0.00000000000001)) ==
              expected_neuron_data[solution.neuron_number() -
                                   solution.output_neuron_number() + i]);
      REQUIRE(
          Catch::Approx(network_output_vector[i]).epsilon((0.00000000000001)) ==
          expected_neuron_data[solution.neuron_number() -
                               solution.output_neuron_number() + i]);
    }
  }
}

TEST_CASE("Solution solver manual testing", "[solve][small][manual-solve]") {
  test_solution_solver_multithread(1);
  test_solution_solver_multithread(2);
  test_solution_solver_multithread(10);
}

/*###############################################################################################
 * Testing if the solution solver produces a correct output, given a built
 * @RafkoNet
 */
void testing_solution_solver_manually(google::protobuf::Arena *arena) {
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
                                                .set_max_solve_threads(4)
                                                .set_device_max_megabytes(2048)
                                                .set_arena_ptr(arena);
  std::vector<std::uint32_t> net_structure = {20, 40, 30, 10, 20};
  std::vector<double> net_input = {10.0, 20.0, 30.0, 40.0, 50.0};

  /* Build the described net and generate a solution from it */
  rafko_net::RafkoNet *net = rafko_net::RafkoNetBuilder(settings)
                                 .input_size(5u)
                                 .expected_input_range((5.0))
                                 .create_layers(net_structure);
  rafko_net::Solution *solution =
      rafko_net::SolutionBuilder(settings).build(*net);

  /* Verify if a generated solution gives back the exact same result, as the
   * manually calculated one */
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      std::make_unique<rafko_net::SolutionSolver>(solution, settings);
  rafko_utilities::ConstVectorSubrange<> neuron_data =
      solver->solve(net_input, true);
  REQUIRE(neuron_data.size() >= solution->output_neuron_number());
  std::vector<double> result = {
      neuron_data.end() - solution->output_neuron_number(), neuron_data.end()};
  std::vector<double> expected_neuron_data =
      std::vector<double>(net->neuron_array_size());
  rafko_test::manaual_fully_connected_network_result(
      net_input, {}, expected_neuron_data, net_structure, *net);
  std::vector<double> expected_result = {expected_neuron_data.end() -
                                             net->output_neuron_number(),
                                         expected_neuron_data.end()};

  /* Verify if the calculated values match the expected ones */
  REQUIRE(net_structure.back() == result.size());
  REQUIRE(expected_result.size() == result.size());
  for (std::uint32_t result_iterator = 0;
       result_iterator < expected_result.size(); ++result_iterator)
    REQUIRE(
        Catch::Approx(result[result_iterator]).epsilon((0.00000000000001)) ==
        expected_result[result_iterator]);

  /* Re-veriy with guaranteed multiple partial solutions */
  double solution_size_mb =
      solution->SpaceUsedLong() /* Bytes */ * 1024.0 /* KB */ * 1024.0 /* MB */;
  (void)settings.set_device_max_megabytes(solution_size_mb / 4.0);
  rafko_net::Solution *solution2 =
      rafko_net::SolutionBuilder(settings).build(*net);

  std::shared_ptr<rafko_net::SolutionSolver> solver2 =
      std::make_unique<rafko_net::SolutionSolver>(solution2, settings);
  rafko_utilities::ConstVectorSubrange<> neuron_data2 =
      solver2->solve(net_input, true);
  REQUIRE(neuron_data2.size() >= solution2->output_neuron_number());
  result = {neuron_data2.end() - solution2->output_neuron_number(),
            neuron_data2.end()};

  /* Verify once more if the calculated values match the expected ones */
  for (std::uint32_t result_iterator = 0;
       result_iterator < expected_result.size(); ++result_iterator)
    REQUIRE(
        Catch::Approx(result[result_iterator]).epsilon((0.00000000000001)) ==
        expected_result[result_iterator]);

  if (nullptr == arena) {
    delete net;
    delete solution;
    delete solution2;
  }
}

TEST_CASE("Solution Solver test based on Fully Connected Dense Net",
          "[solve][build-solve]") {
  testing_solution_solver_manually(nullptr);
}

/*###############################################################################################
 * Testing if the solution solver produces correct data for Networks generated
 * with connections of memories of the past
 *//* The utility function returns with the number of megabytes required for the complete Solution */
double testing_nets_with_memory_manually(google::protobuf::Arena *arena,
                                         double max_space_mb, bool recursion,
                                         bool boltzman_knot) {
  std::vector<std::uint32_t> net_structure = {20, 30, 40, 30, 20};
  std::vector<double> net_input = {10.0, 20.0, 30.0, 40.0, 50.0};

  /* Build the above described net */
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings()
          .set_arena_ptr(arena)
          .set_device_max_megabytes(max_space_mb);
  rafko_net::RafkoNetBuilder net_builder = rafko_net::RafkoNetBuilder(settings);
  net_builder.input_size(5).expected_input_range((5.0));

  /* Add inputs from the past */
  std::uint32_t layer_index = rand() % net_structure.size();
  if (recursion)
    net_builder.add_neuron_recurrence(layer_index,
                                      rand() % net_structure[layer_index], 1u);
  if (boltzman_knot)
    net_builder.add_feature_to_layer(
        layer_index, rafko_net::neuron_group_feature_boltzmann_knot);

  rafko_net::RafkoNet *net = net_builder.create_layers(net_structure);

  /* Generate solution from Net */
  rafko_net::Solution *solution =
      rafko_net::SolutionBuilder(settings).build(*net);
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      std::make_unique<rafko_net::SolutionSolver>(solution, settings);

  /* Verify if a generated solution gives back the exact same result, as the
   * manually calculated one */
  rafko_utilities::ConstVectorSubrange<> neuron_data =
      solver->solve(net_input, true);
  REQUIRE(neuron_data.size() >= solution->output_neuron_number());
  std::vector<double> result = {
      (neuron_data.end() - solution->output_neuron_number()),
      neuron_data.end()};
  std::vector<double> previous_neuron_data =
      std::vector<double>(net->neuron_array_size());
  std::vector<double> expected_neuron_data = std::vector<double>(
      net->neuron_array_size()); /* Should be all zeroes the first time */

  rafko_test::manaual_fully_connected_network_result(
      net_input, previous_neuron_data, expected_neuron_data, net_structure,
      *net);
  std::vector<double> expected_result = {expected_neuron_data.end() -
                                             net->output_neuron_number(),
                                         expected_neuron_data.end()};

  REQUIRE(net_structure.back() == result.size());
  REQUIRE(expected_result.size() == result.size());
  for (std::uint32_t result_iterator = 0;
       result_iterator < expected_result.size(); ++result_iterator) {
    CHECK(Catch::Approx(result[result_iterator]).epsilon((0.00000000000001)) ==
          expected_result[result_iterator]);
  }

  for (std::uint32_t loop = 0; loop < 5;
       ++loop) { /* Re-verify with additional runs, at least 3, more shouldn't
                    hurt */
    rafko_utilities::ConstVectorSubrange<> neuron_data =
        solver->solve(net_input, false);
    REQUIRE(neuron_data.size() >= solution->output_neuron_number());
    result = {neuron_data.end() - solution->output_neuron_number(),
              neuron_data.end()};
    previous_neuron_data = std::vector<double>(expected_neuron_data);
    rafko_test::manaual_fully_connected_network_result(
        net_input, previous_neuron_data, expected_neuron_data, net_structure,
        *net);
    expected_result = {expected_neuron_data.end() - net->output_neuron_number(),
                       expected_neuron_data.end()};

    REQUIRE(net_structure.back() == result.size());
    REQUIRE(expected_result.size() == result.size());
    for (std::uint32_t result_iterator = 0;
         result_iterator < expected_result.size(); ++result_iterator)
      REQUIRE(
          Catch::Approx(result[result_iterator]).epsilon((0.00000000000001)) ==
          expected_result[result_iterator]);
  }

  /* Return with the size of the overall solution */
  double space_used_mb =
      solution->SpaceUsedLong() /* Bytes */ * 1024.0 /* KB */ * 1024.0 /* MB */;

  if (nullptr == settings.get_arena_ptr()) {
    delete net;
    delete solution;
  }

  return space_used_mb;
}

TEST_CASE("Solution Solver test with memory", "[solve][memory]") {
  /* Test if the network is producing correct results when neurons take
   * past-inputs from themselves ( 0x01 ID given to builder ) */
  double megabytes_used =
      testing_nets_with_memory_manually(nullptr, (4.0 * 1024.0), true, false);
  (void)testing_nets_with_memory_manually(
      nullptr, megabytes_used / 4.0, true,
      false); /* Even if the net needs to be splitted */

  /* Test if the network is producing correct results when neurons take
   * past-inputs from their layers ( 0x02 ID given to builder ) */
  megabytes_used =
      testing_nets_with_memory_manually(nullptr, (4.0 * 1024.0), true, true);
  (void)testing_nets_with_memory_manually(
      nullptr, megabytes_used / 4.0, true,
      true); /* Even if the net needs to be splitted */
}

/*###############################################################################################
 * Calculate a generated Fully Connected dense network manually by the network
 * description and compare the calculated results to the one provided by the
 * solution.
 */
void test_generated_net_by_calculation(google::protobuf::Arena *arena) {
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_arena_ptr(arena);
  std::vector<double> net_input = {10.0, 20.0, 30.0, 40.0, 50.0};
  std::vector<std::uint32_t> network_layout_sizes = {10, 30, 20};

  /* Generate a fully connected Neural network */
  std::unique_ptr<rafko_net::RafkoNetBuilder> builder(
      std::make_unique<rafko_net::RafkoNetBuilder>(settings));
  builder->input_size(5)
      .output_neuron_number(network_layout_sizes.back())
      .expected_input_range((5.0));

  rafko_net::RafkoNet *network(builder->create_layers(
      network_layout_sizes,
      {{rafko_net::transfer_function_identity},
       {rafko_net::transfer_function_selu, rafko_net::transfer_function_relu},
       {rafko_net::transfer_function_tanh,
        rafko_net::transfer_function_sigmoid}}));

  /* Generate a solution */
  rafko_net::Solution *solution = nullptr;
  REQUIRE_NOTHROW(solution =
                      rafko_net::SolutionBuilder(settings).build(*network));
  settings.set_device_max_megabytes(/* Introduce segmentation into the solution
                                       to test roboustness */
                                    (solution->SpaceUsedLong() /* Bytes */ /
                                     1024.0 /* KB */ / 1024.0 /* MB */) /
                                    4.0);
  REQUIRE_NOTHROW(solution =
                      rafko_net::SolutionBuilder(settings).build(*network));

  /* Solve the generated solution */
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      std::make_unique<rafko_net::SolutionSolver>(solution, settings);

  /* Verify if a generated solution gives back the exact same result, as the
   * manually calculated one */
  rafko_utilities::ConstVectorSubrange<> network_output =
      solver->solve(net_input, true);

  /* Calculate the network manually */
  rafko_net::TransferFunction transfer_function(settings);
  const std::uint32_t number_of_neurons = std::accumulate(
      network_layout_sizes.begin(), network_layout_sizes.end(), 0);
  std::vector<double> manual_neuron_values =
      std::vector<double>(number_of_neurons);
  std::vector<bool> solved = std::vector<bool>(number_of_neurons, false);
  std::uint32_t solved_neurons = 0u;
  std::uint32_t solved_neurons_in_loop = -1;
  std::uint32_t solved_inputs_in_neuron;
  std::uint32_t overall_inputs_in_neuron;
  std::int32_t input_index;
  double neuron_data;
  double spike_function_weight;
  std::uint32_t neuron_input_iterator = 0;
  bool first_weight_in_synapse;
  while (
      (number_of_neurons >
       solved_neurons) /* Until all of the Neurons are solved */
      &&
      (0 < solved_neurons_in_loop) /* but in case no neurons could be solved in
                                      this loop, infinite loop is suspected */
  ) {
    solved_neurons_in_loop = 0;
    /* Go for each neuron */
    for (std::uint32_t neuron_iterator = 0; neuron_iterator < number_of_neurons;
         ++neuron_iterator) {
      /* if the Neuron is solvable --> all of its children are etiher inputs or
       * solved already */
      /* solve them, store its data and update the meta */
      if (false == solved[neuron_iterator]) {
        rafko_net::SynapseIterator<rafko_net::InputSynapseInterval>
            neuron_input_synapses(
                network->neuron_array(neuron_iterator).input_indices());
        overall_inputs_in_neuron = neuron_input_synapses.size();
        solved_inputs_in_neuron = 0;
        neuron_input_iterator = 0;
        neuron_data = 0;
        first_weight_in_synapse = true;
        spike_function_weight = (0.0);
        rafko_net::SynapseIterator<>::iterate(
            network->neuron_array(neuron_iterator).input_weights(),
            [&](std::int32_t weight_index) {
              if (true == first_weight_in_synapse) {
                first_weight_in_synapse = false;
                spike_function_weight = network->weight_table(weight_index);
              } else {
                if (neuron_input_iterator < neuron_input_synapses.size()) {
                  input_index = neuron_input_synapses[neuron_input_iterator];
                  if (rafko_net::SynapseIterator<>::is_index_input(
                          input_index) /* Neuron input points to input data */
                      ||
                      (true ==
                       solved[input_index]) /* or the current input points to a
                                               neuron which is already solved */
                  ) {                       /* the input counts as solved */
                    ++solved_inputs_in_neuron;
                  }
                  if (rafko_net::SynapseIterator<>::is_index_input(
                          input_index)) {
                    input_index = rafko_net::SynapseIterator<>::
                        array_index_from_external_index(input_index);
                    neuron_data += net_input[input_index] *
                                   network->weight_table(weight_index);
                  } else {
                    neuron_data += manual_neuron_values[input_index] *
                                   network->weight_table(weight_index);
                  }
                  ++neuron_input_iterator;
                } else { /* After the inputs, every weight is the bias */
                  neuron_data += network->weight_table(weight_index);
                }
              }
            });
        if (solved_inputs_in_neuron == overall_inputs_in_neuron) {
          neuron_data = transfer_function.get_value(
              network->neuron_array(neuron_iterator).transfer_function(),
              neuron_data);
          manual_neuron_values[neuron_iterator] =
              rafko_net::SpikeFunction::get_value(
                  network->neuron_array(neuron_iterator).spike_function(),
                  spike_function_weight, neuron_data,
                  manual_neuron_values[neuron_iterator]);
          solved[neuron_iterator] = true;
          ++solved_neurons;
          ++solved_neurons_in_loop;
        }
      } /*(false == solved[neuron_iterator])*/ /* if the condition is false, it
                                                  means the neuron is already
                                                  solved */
    }

  } /*while(the neurons are solved)*/
  REQUIRE(number_of_neurons == solved_neurons);

  /* Compare the calculated Neuron outputs to the values in the solution */
  for (std::uint32_t neuron_index = 0;
       neuron_index < network_layout_sizes.back(); ++neuron_index) {
    REQUIRE(/* Solution solver only provides the data of the output neurons! */
            manual_neuron_values[number_of_neurons -
                                 network_layout_sizes.back() + neuron_index] ==
            network_output[neuron_index]);
  }
  if (nullptr == arena) {
    delete network;
    delete solution;
  }
}

TEST_CASE("Solution Solver test with Generated fully connected network",
          "[solve][full]") {
  test_generated_net_by_calculation(nullptr);
}

/*###############################################################################################
 * Test if the solver is able to produce correct output when used from multiple
 * threads
 */
TEST_CASE("Solution Solver Multi-threading test",
          "[solve][full][multithread]") {
  google::protobuf::Arena arena;
  std::vector<std::uint32_t> net_structure = {20, 30, 40, 30, 20};
  std::vector<double> net_input = {10.0, 20.0, 30.0, 40.0, 50.0};
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_arena_ptr(&arena);
  rafko_net::RafkoNet &network = *rafko_net::RafkoNetBuilder(settings)
                                      .input_size(5)
                                      .expected_input_range((5.0))
                                      .create_layers(net_structure);
  rafko_net::Solution *solution =
      rafko_net::SolutionBuilder(settings).build(network);
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      std::make_unique<rafko_net::SolutionSolver>(solution, settings);

  /* solve in a single thread */
  rafko_utilities::ConstVectorSubrange<> single_thread_output_buffer =
      solver->solve(net_input, true);
  const std::vector<double> single_thread_output = {
      single_thread_output_buffer.begin(), single_thread_output_buffer.end()};

  /* solve from multiple threads */
  const std::uint32_t thread_number = settings.get_max_processing_threads();
  rafko_utilities::ThreadGroup executor(thread_number);
  std::vector<std::vector<double>> thread_outputs(thread_number);
  executor.start_and_block([&](std::uint32_t thread_index) {
    rafko_utilities::ConstVectorSubrange<> thread_output_buffer =
        solver->solve(net_input, true, thread_index);
    thread_outputs[thread_index] = {thread_output_buffer.begin(),
                                    thread_output_buffer.end()};
  });

  /* compare that multi-thread solve should be the same as single thread solve
   */
  for (std::uint32_t neuron_data_index = 0;
       neuron_data_index < single_thread_output.size(); ++neuron_data_index) {
    for (std::uint32_t thread_index = 0; thread_index < thread_number;
         ++thread_index) {
      REQUIRE(single_thread_output[neuron_data_index] ==
              thread_outputs[thread_index][neuron_data_index]);
    }
  }
}

/*###############################################################################################
 * Test if the solver is able to remember the previous neuron values correctly
 */
TEST_CASE("Solution Solver memory test", "[solve][memory]") {
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_arena_ptr(&arena);
  rafko_net::RafkoNet &net(*rafko_net::RafkoNetBuilder(settings)
                                .input_size(1)
                                .expected_input_range((5.0))
                                .add_neuron_recurrence(0u, 0u, 1u)
                                .allowed_transfer_functions_by_layer(
                                    {{rafko_net::transfer_function_identity}})
                                .create_layers({1}));

  net.set_weight_table(0u,
                       (0.0)); /* Set the memory filter of the only neuron to 0,
                                  so the previous value of it would not modify
                                  the current one through the spike function */
  for (std::int32_t weight_index = 1; weight_index < net.weight_table_size();
       ++weight_index) {
    net.set_weight_table(weight_index, (1.0));
  }

  rafko_net::Solution *solution(
      rafko_net::SolutionBuilder(settings).build(net));
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      std::make_unique<rafko_net::SolutionSolver>(solution, settings);

  double expected_result = (1.0);
  for (std::uint32_t variant = 0u; variant < 10u; ++variant) {
    CHECK(expected_result == (solver->solve({(0.0)}, false, 0u))[0]);
    expected_result += (1.0);
  }
}

TEST_CASE("Solution Solver Neuron benchmark", "[runtime][!benchmark]") {
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings =
      std::make_shared<rafko_mainframe::RafkoSettings>(
          rafko_mainframe::RafkoSettings().set_arena_ptr(&arena));
  rafko_net::RafkoNet &network(*rafko_net::RafkoNetBuilder(*settings)
                                    .input_size(1)
                                    .expected_input_range((5.0))
                                    .add_neuron_recurrence(0u, 0u, 1u)
                                    .create_layers({10, 20}));
  std::chrono::steady_clock::time_point start;
  start = std::chrono::steady_clock::now();
  std::shared_ptr<rafko_net::SolutionSolver> solver =
      rafko_net::SolutionSolver::Factory(network, settings).build();
  std::cout << "creation duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count()
            << ";" << std::endl;

  std::uint32_t avg_ms = 0;
  while (true) {
    start = std::chrono::steady_clock::now();
    solver->solve({0});
    auto current_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    std::cout << "\rrun duration: " << current_duration
              << "ms; \t\tavg:" << avg_ms << "ms      ";
    avg_ms = (current_duration + avg_ms) / 2;
  }
}

} // namespace rafko_net_test
