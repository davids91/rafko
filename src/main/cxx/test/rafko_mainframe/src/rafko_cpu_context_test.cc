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

#include <vector>
#include <memory>
#include <functional>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/services/cost_function_mse.h"
#include "rafko_gym/models/rafko_cost.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_cpu_context.h"

#include "test/test_utility.h"

namespace rafko_gym_test {

TEST_CASE("Testing if CPU context produces correct error values upon full evaluation", "[context][CPU][evaluation]"){
  std::uint32_t sample_number = 50;
  std::uint32_t sequence_size = 6;
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  double expected_label = (50.0);
  rafko_net::RafkoNet* network = rafko_test::generate_random_net_with_softmax_features(1u, settings);
  std::unique_ptr<rafko_gym::DataSet> dataset(rafko_test::create_dataset(network->input_data_size(), network->output_neuron_number(), sample_number, sequence_size, 0/*prefill_size*/, expected_label));
  std::shared_ptr<rafko_gym::CostFunction> cost = std::make_shared<rafko_gym::CostFunctionMSE>(settings);
  rafko_gym::RafkoDatasetWrapper dataset_wrap(*dataset);
  rafko_gym::RafkoCost reference_cost(settings, cost);
  rafko_mainframe::RafkoCPUContext context(*network, settings);
  context.set_objective(std::make_unique<rafko_gym::RafkoCost>(settings, cost));
  context.set_environment(std::make_unique<rafko_gym::RafkoDatasetWrapper>(*dataset));

  /* Set some error and see if the environment produces the expected */
  std::unique_ptr<rafko_net::Solution> solution = rafko_net::SolutionBuilder(settings).build(*network);
  std::unique_ptr<rafko_net::SolutionSolver> reference_solver(rafko_net::SolutionSolver::Builder(*solution, settings).build());

  double error_sum = (0.0);
  std::uint32_t raw_inputs_index = 0;
  std::uint32_t raw_label_index = 0;
  reference_solver->set_eval_mode(true);
  for(std::uint32_t sequence_index = 0; sequence_index < dataset_wrap.get_number_of_sequences(); ++sequence_index){
    bool reset = true;
    for(std::uint32_t prefill_index = 0; prefill_index < dataset_wrap.get_prefill_inputs_number(); ++prefill_index){
      (void)reference_solver->solve(dataset_wrap.get_input_sample(raw_inputs_index), reset);
      if(reset)reset = false;
      ++raw_inputs_index;
    }
    for(std::uint32_t label_inside_sequence = 0; label_inside_sequence < dataset_wrap.get_sequence_size(); ++label_inside_sequence){
      rafko_utilities::ConstVectorSubrange<> neuron_output = reference_solver->solve(dataset_wrap.get_input_sample(raw_inputs_index), reset);
      double err = reference_cost.set_feature_for_label( dataset_wrap, raw_label_index, {neuron_output.begin(),neuron_output.end()} );
      error_sum += err;
      if(reset)reset = false;
      ++raw_inputs_index;
      ++raw_label_index;
    }
  }
  double environment_error = context.full_evaluation();
  CHECK( Catch::Approx(environment_error).margin(0.00000000000001) == -(error_sum / (sample_number * sequence_size)) );
}

TEST_CASE("Testing if CPU context produces correct error values upon stochastic evaluation", "[context][CPU][evaluation]"){
  std::uint32_t seed = rand() + 1;
  std::uint32_t sample_number = 50;
  std::uint32_t sequence_size = 6;
  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  double expected_label = (50.0);
  rafko_net::RafkoNet* network = rafko_test::generate_random_net_with_softmax_features(1u, settings);
  std::unique_ptr<rafko_gym::DataSet> dataset(rafko_test::create_dataset(network->input_data_size(), network->output_neuron_number(), sample_number, sequence_size, 0/*prefill_size*/, expected_label));
  std::shared_ptr<rafko_gym::CostFunction> cost = std::make_shared<rafko_gym::CostFunctionMSE>(settings);
  rafko_gym::RafkoDatasetWrapper dataset_wrap(*dataset);
  rafko_gym::RafkoCost reference_cost(settings, cost);
  rafko_mainframe::RafkoCPUContext context(*network, settings);

  (void)context.expose_settings().set_memory_truncation(dataset_wrap.get_sequence_size() / 2); /*!Note: So overall error value can just be halved because of this */
  settings = context.expose_settings();

  context.set_objective(std::make_unique<rafko_gym::RafkoCost>(settings, cost));
  context.set_environment(std::make_unique<rafko_gym::RafkoDatasetWrapper>(*dataset));

  double environment_error = context.stochastic_evaluation(true, seed);

  std::unique_ptr<rafko_net::Solution> solution = rafko_net::SolutionBuilder(settings).build(*network);
  std::unique_ptr<rafko_net::SolutionSolver> reference_solver(rafko_net::SolutionSolver::Builder(*solution, settings).build());

  srand(seed);
  double error_sum = (0.0);
  std::uint32_t sequence_start_index = (rand()%(dataset_wrap.get_number_of_sequences() - settings.get_minibatch_size() + 1));
  std::uint32_t start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
    dataset_wrap.get_sequence_size() - settings.get_memory_truncation() + 1 /* ..not all result output values are evaluated.. */
  )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */
  std::uint32_t raw_inputs_index = 0;
  std::uint32_t raw_label_index = 0;
  reference_solver->set_eval_mode(true);
  for(std::uint32_t sequence_index = sequence_start_index; sequence_index < (sequence_start_index + settings.get_minibatch_size()); ++sequence_index){
    bool reset = true;
    for(std::uint32_t prefill_index = 0; prefill_index < dataset_wrap.get_prefill_inputs_number(); ++prefill_index){
      (void)reference_solver->solve(dataset_wrap.get_input_sample(raw_inputs_index), reset);
      if(reset)reset = false;
      ++raw_inputs_index;
    }
    for(std::uint32_t label_inside_sequence = 0; label_inside_sequence < dataset_wrap.get_sequence_size(); ++label_inside_sequence){
      rafko_utilities::ConstVectorSubrange<> neuron_output = reference_solver->solve(dataset_wrap.get_input_sample(raw_inputs_index), reset);
      double err = reference_cost.set_feature_for_label( dataset_wrap, raw_label_index, {neuron_output.begin(),neuron_output.end()} );
      if(
        (label_inside_sequence >= start_index_inside_sequence)
        &&(label_inside_sequence < (start_index_inside_sequence + settings.get_memory_truncation()))
      ){
        error_sum += err;
      }
      if(reset)reset = false;
      ++raw_inputs_index;
      ++raw_label_index;
    }
  }
  CHECK( Catch::Approx(environment_error).margin(0.00000000000001)
    == -( error_sum / static_cast<double>(settings.get_minibatch_size() * sequence_size) )
  );
}

TEST_CASE("Testing if CPU context updates weights according to requests", "[context]"){
}

} /* namespace rako_gym_test */
