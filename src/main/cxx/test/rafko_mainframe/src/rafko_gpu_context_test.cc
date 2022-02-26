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
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_cost.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_mainframe/services/rafko_gpu_context.h"
#include "rafko_mainframe/services/rafko_cpu_context.h"
#include "test/test_utility.h"

namespace rafko_gym_test {

TEST_CASE("Testing if GPU Context is able to build a valid openCL environment", "[context][GPU]"){
  rafko_mainframe::RafkoSettings settings;
  rafko_net::RafkoNet* network = rafko_test::generate_random_net_with_softmax_features(1u, settings);
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  REQUIRE_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );
}

TEST_CASE("Testing if standalone solution is working as intended with the GPU context","[context][GPU][solve]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 50u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,2});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    std::unique_ptr<rafko_net::Solution> reference_solution = rafko_net::SolutionBuilder(settings).build(*network);
    std::unique_ptr<rafko_net::SolutionSolver> reference_agent = rafko_net::SolutionSolver::Builder(*reference_solution, settings).build();
    std::vector<double> network_input(network->input_data_size(), (rand()%10));
    rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
    rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);

    reference_agent->set_eval_mode(false);
    for(std::uint32_t result_index = 0; result_index < reference_result.size(); ++result_index){
      CHECK( Catch::Approx(reference_result[result_index]).epsilon(0.0000000001) == context_result[result_index] );
    }
  }/*for(50 variants)*/
}

TEST_CASE("Testing if standalone solution is working as intended with the GPU context even with softmax features","[context][GPU][features][softmax]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 50u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      )
      .add_feature_to_layer(2, rafko_net::neuron_group_feature_softmax)
      .add_feature_to_layer(3, rafko_net::neuron_group_feature_softmax)
      .add_feature_to_layer(4, rafko_net::neuron_group_feature_softmax)
      .dense_layers({2,2,2,2,2,2});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    std::unique_ptr<rafko_net::Solution> reference_solution = rafko_net::SolutionBuilder(settings).build(*network);
    std::unique_ptr<rafko_net::SolutionSolver> reference_agent = rafko_net::SolutionSolver::Builder(*reference_solution, settings).build();
    std::vector<double> network_input(network->input_data_size(), (rand()%10));
    rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
    rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);

    reference_agent->set_eval_mode(false);
    for(std::uint32_t result_index = 0; result_index < reference_result.size(); ++result_index){
      CHECK( Catch::Approx(reference_result[result_index]).epsilon(0.0000000001) == context_result[result_index] );
    }
  }/*for(50 variants)*/
}

TEST_CASE("Testing if a standalone solution is working as intended with the GPU context even with inputs from the past","[context][GPU][solve][memory]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .set_recurrence_to_layer()
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,2});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    std::unique_ptr<rafko_net::Solution> reference_solution = rafko_net::SolutionBuilder(settings).build(*network);
    std::unique_ptr<rafko_net::SolutionSolver> reference_agent = rafko_net::SolutionSolver::Builder(*reference_solution, settings).build();
    std::vector<double> network_input(network->input_data_size(), (rand()%10));

    reference_agent->set_eval_mode(false);
    for(std::uint32_t steps = 0; steps < 5; ++steps){
      rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
      rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);
      for(std::uint32_t result_index = 0; result_index < reference_result.size(); ++result_index){
        CHECK( Catch::Approx(reference_result[result_index]).epsilon(0.0000000001) == context_result[result_index] );
      }
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with single sample of sequence size one","[context][GPU][evaluate]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = 1;
  std::uint32_t number_of_sequences = 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,1});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    rafko_mainframe::RafkoCPUContext reference_context(*network, settings);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(std::uint32_t steps = 0; steps < 1; ++steps){
      std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<double>>(std::get<0>(tmp1)),
        std::vector<std::vector<double>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(std::uint32_t i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with single sample of sequence size one with recurrence in the network","[context][GPU][evaluate]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .set_recurrence_to_layer()
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,1});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    rafko_mainframe::RafkoCPUContext reference_context(*network, settings);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(std::uint32_t steps = 0; steps < 1; ++steps){
      std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<double>>(std::get<0>(tmp1)),
        std::vector<std::vector<double>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(std::uint32_t i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels","[context][GPU][evaluate][multi-label]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;

  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .set_recurrence_to_layer()
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,1});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    rafko_mainframe::RafkoCPUContext reference_context(*network, settings);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_mse
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    Catch::StringMaker<double>::precision = 20;
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 1;
      (void)context->expose_settings().set_memory_truncation(sequence_size);
      (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
      std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<double>>(std::get<0>(tmp1)),
        std::vector<std::vector<double>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(std::uint32_t i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels and sequential data","[context][GPU][evaluate][multi-label][sequence]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;

  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .set_recurrence_to_layer()
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,1});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    rafko_mainframe::RafkoCPUContext reference_context(*network, settings);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_cross_entropy
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 1;
      sequence_size = rand()%10 + 1;
      (void)context->expose_settings().set_memory_truncation(sequence_size);
      (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
      std::pair<std::vector<std::vector<double>>,std::vector<std::vector<double>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<double>>(std::get<0>(tmp1)),
        std::vector<std::vector<double>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(std::uint32_t i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels and sequential data and prefill","[context][GPU][evaluate][multi-label][sequence][prefill]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;
  std::uint32_t feature_size = rand()%5 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range((1.0))
      .set_recurrence_to_layer()
      .allowed_transfer_functions_by_layer(
        {
          {rafko_net::transfer_function_identity},
          {rafko_net::transfer_function_sigmoid},
          {rafko_net::transfer_function_tanh},
          {rafko_net::transfer_function_elu},
          {rafko_net::transfer_function_selu},
          {rafko_net::transfer_function_relu},
        }
      ).dense_layers({2,2,2,2,2,feature_size});
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
    CHECK_NOTHROW(
      context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    rafko_mainframe::RafkoCPUContext reference_context(*network, settings);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_binary_cross_entropy
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );
    /*!Note: if 15 digits are used for comparison instead of 10 sometimes there's a mismatch */

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 1;
      sequence_size = rand()%10 + 1;
      (void)context->expose_settings().set_memory_truncation(sequence_size);
      (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
      std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
        2/* input size */, feature_size,
        number_of_sequences, sequence_size, 2/*prefill_size*/,
        rand()%100/*expected_label*/
      ) );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        *dataset
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(std::uint32_t i = 0; i < 3; ++i){
        (void)context->stochastic_evaluation(); /* to fill up buffers with something */
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
      }
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

void preapare_eval_buffers_for_seed(
  std::uint32_t seed,
  std::vector<std::vector<double>>& reference_inputs,
  std::vector<std::vector<double>>& reference_features,
  std::vector<std::vector<double>>& reference_labels,
  std::vector<std::uint32_t>& reference_sequence_index_values,
  std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment,
  rafko_mainframe::RafkoSettings& settings,
  rafko_net::SolutionSolver& reference_agent,
  std::uint32_t& used_minibatch_size,
  std::uint32_t& start_index_in_sequence,
  std::uint32_t& used_sequence_truncation
){
  used_minibatch_size = std::min(settings.get_minibatch_size(), environment->get_number_of_sequences());
  reference_inputs.resize(
    used_minibatch_size * (environment->get_sequence_size() + environment->get_prefill_inputs_number())
  );
  reference_features.resize(
    used_minibatch_size * environment->get_sequence_size()
  );
  reference_labels.resize(
    used_minibatch_size * environment->get_sequence_size()
  );
  reference_sequence_index_values.resize(used_minibatch_size);

  srand(seed);
  std::uint32_t uploaded_sequences = 0u;
  const std::uint32_t inputs_in_a_sequence = environment->get_sequence_size() + environment->get_prefill_inputs_number();
  const std::uint32_t labels_in_a_sequence = environment->get_sequence_size();
  used_sequence_truncation = std::min(
    settings.get_memory_truncation(), environment->get_sequence_size()
  );
  start_index_in_sequence = (rand()%(
    environment->get_sequence_size() - used_sequence_truncation + 1
  ));
  while(uploaded_sequences < used_minibatch_size){
    std::uint32_t sequences_to_upload = rand()%(used_minibatch_size - uploaded_sequences + 1u);
    std::uint32_t sequence_start_index = rand()%(environment->get_number_of_sequences() - sequences_to_upload + 1u);

    for(std::uint32_t s = 0; s < sequences_to_upload; ++s){
      reference_sequence_index_values[uploaded_sequences + s] = sequence_start_index + s;
    }

    /* copy inputs */
    std::uint32_t raw_input_start = (sequence_start_index * inputs_in_a_sequence);
    std::uint32_t raw_input_num = (sequences_to_upload * inputs_in_a_sequence);
    std::uint32_t reference_offset = (uploaded_sequences * inputs_in_a_sequence);
    for(std::uint32_t raw_input_index = raw_input_start; raw_input_index < (raw_input_start + raw_input_num); ++raw_input_index){
      reference_inputs[reference_offset] = environment->get_input_sample(raw_input_index);
      ++reference_offset;
    }

    /* copy labels */
    std::uint32_t raw_label_start = (sequence_start_index * labels_in_a_sequence);
    std::uint32_t raw_label_num = (sequences_to_upload * labels_in_a_sequence);
    reference_offset = (uploaded_sequences * labels_in_a_sequence);

    for(std::uint32_t raw_label_index = raw_label_start; raw_label_index < (raw_label_start + raw_label_num); ++raw_label_index){
      reference_labels[reference_offset] = environment->get_label_sample(raw_label_index);
      ++reference_offset;
    }

    uploaded_sequences += sequences_to_upload;
  }/*while(minibatch is filled with random data)*/

  /* generate features */
  std::uint32_t sequences_done = 0u;
  std::uint32_t reference_offset = 0u;
  while(sequences_done < used_minibatch_size){
    std::uint32_t raw_inputs_index = ( reference_sequence_index_values[sequences_done] * inputs_in_a_sequence );

    for(std::uint32_t prefill_iterator = 0; prefill_iterator < environment->get_prefill_inputs_number(); ++prefill_iterator){
      (void)reference_agent.solve(
        environment->get_input_sample(raw_inputs_index),
        (0 == prefill_iterator)/*reset_neuron_data*/
      );
      ++raw_inputs_index;
    }

    for(std::uint32_t sequence_iterator = 0; sequence_iterator < environment->get_sequence_size(); ++sequence_iterator){
      rafko_utilities::ConstVectorSubrange<> neuron_output = reference_agent.solve(
        environment->get_input_sample(raw_inputs_index),
        ( (0u == environment->get_prefill_inputs_number())&&(0u == sequence_iterator) )/*reset_neuron_data*/
      );
      reference_features[reference_offset] = std::vector<double>{ neuron_output.begin(), neuron_output.end() };
      ++reference_offset;
      ++raw_inputs_index;
    }
    ++sequences_done;
  }
}

TEST_CASE("Testing Stochastic evaluation with the GPU context","[stochastic][context][GPU][evaluate][multi-label][sequence][prefill]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;
  std::uint32_t feature_size = rand()%5 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
    .input_size(2).expected_input_range((1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
      {
        {rafko_net::transfer_function_identity},
        {rafko_net::transfer_function_sigmoid},
        {rafko_net::transfer_function_tanh},
        {rafko_net::transfer_function_elu},
        {rafko_net::transfer_function_selu},
        {rafko_net::transfer_function_relu},
      }
    ).dense_layers({2,2,2,2,2,feature_size});
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  rafko_net::RafkoNet network_copy = rafko_net::RafkoNet(*network);
  CHECK_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );
  std::unique_ptr<rafko_net::Solution> solution = rafko_net::SolutionBuilder(settings).build(network_copy);
  std::unique_ptr<rafko_net::SolutionSolver> reference_agent(rafko_net::SolutionSolver::Builder(*solution, settings).build());

  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_squared_error
    );
    std::unique_ptr<rafko_gym::CostFunction> cost_sqr_error = rafko_gym::FunctionFactory::build_cost_function(
      rafko_gym::cost_function_squared_error, settings
    );

    context->set_objective(objective);

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 2;
      sequence_size = rand()%10 + 2;
      settings = context->expose_settings().set_memory_truncation(sequence_size);
      std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
        2/* input size */, feature_size,
        number_of_sequences, sequence_size, 2/*prefill_size*/,
        rand()%100/*expected_label*/, (1.0)
      ) );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(*dataset);
      std::vector<std::vector<double>> reference_inputs;
      std::vector<std::vector<double>> reference_features;
      std::vector<std::vector<double>> reference_labels;
      std::vector<std::uint32_t> reference_sequence_index_values;

      std::uint32_t used_minibatch_size;
      std::uint32_t start_index_in_sequence;
      std::uint32_t used_sequence_truncation;
      context->set_environment(environment);
      /* upload random labels and inputs */
      std::uint32_t seed = rand();
      preapare_eval_buffers_for_seed(
        seed, reference_inputs, reference_features, reference_labels, reference_sequence_index_values,
        environment, settings, *reference_agent, used_minibatch_size, start_index_in_sequence, used_sequence_truncation
      );

      double minibatch_error = 0u;
      for(std::uint32_t minibatch_index = 0; minibatch_index < used_minibatch_size; ++minibatch_index){
        double partial_sum = objective->set_features_for_sequences(
          *environment, reference_features, (minibatch_index * environment->get_sequence_size())/* neuron_buffer_index */,
          reference_sequence_index_values[minibatch_index]/*sequence_start_index*/, 1u/* sequences_to_evaluate */,
          start_index_in_sequence, used_sequence_truncation
        );
        minibatch_error += partial_sum;
      }

      minibatch_error /= static_cast<double>(used_minibatch_size * environment->get_sequence_size());
      REQUIRE( Catch::Approx(-minibatch_error).epsilon(0.00000000000001) == context->stochastic_evaluation(true, seed) );

      for(std::uint32_t i = 0; i < 5; ++i){

        minibatch_error = 0u;
        for(std::uint32_t minibatch_index = 0; minibatch_index < used_minibatch_size; ++minibatch_index){
          double partial_sum = objective->set_features_for_sequences(
            *environment, reference_features, (minibatch_index * environment->get_sequence_size())/* neuron_buffer_index */,
            reference_sequence_index_values[minibatch_index]/*sequence_start_index*/, 1u/* sequences_to_evaluate */,
            start_index_in_sequence, used_sequence_truncation
          );
          minibatch_error += partial_sum;
        }
        minibatch_error /= static_cast<double>(used_minibatch_size * environment->get_sequence_size());

        (void)context->full_evaluation(); /* to fill up buffers with something else */
        REQUIRE( Catch::Approx(context->stochastic_evaluation(true, seed)).epsilon(0.00000000000001) == context->stochastic_evaluation(true, seed) );
        (void)context->stochastic_evaluation(true, (seed + 1u)); /* to fill up buffers with something else */
        REQUIRE( Catch::Approx(-minibatch_error).epsilon(0.00000000000001) == context->stochastic_evaluation(true, seed) );

        seed = rand();
        settings = context->expose_settings().set_memory_truncation(rand()%sequence_size + 1);
        settings = context->expose_settings().set_minibatch_size(rand()%environment->get_number_of_sequences() + 1);
        preapare_eval_buffers_for_seed(
          seed, reference_inputs, reference_features, reference_labels, reference_sequence_index_values,
          environment, settings, *reference_agent, used_minibatch_size, start_index_in_sequence, used_sequence_truncation
        );
      }/*for(5 inner consecutive steps)*/
    }/*for(5 consecutive steps)*/
  }/*for(10 variants)*/
}


TEST_CASE("Testing weight updates with the GPU context","[context][GPU][weight-update]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;
  std::uint32_t feature_size = rand()%5 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
    .input_size(2).expected_input_range((1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
      {
        {rafko_net::transfer_function_identity},
        {rafko_net::transfer_function_sigmoid},
        {rafko_net::transfer_function_tanh},
        {rafko_net::transfer_function_elu},
        {rafko_net::transfer_function_selu},
        {rafko_net::transfer_function_relu},
      }
    ).dense_layers({2,2,2,2,2,feature_size});
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  CHECK_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );
  rafko_mainframe::RafkoCPUContext reference_context(*network, settings);

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    settings, rafko_gym::cost_function_cross_entropy
  );
  reference_context.set_objective(objective);
  context->set_objective(objective);

  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    number_of_sequences = rand()%10 + 1;
    sequence_size = rand()%10 + 1;
    (void)context->expose_settings().set_memory_truncation(sequence_size);
    (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
    std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
      2/* input size */, feature_size,
      number_of_sequences, sequence_size, 2/*prefill_size*/,
      rand()%100/*expected_label*/, (1.0)
    ) );
    std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(*dataset);

    context->set_environment(environment);
    reference_context.set_environment(environment);

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );

      /* modify single weight */
      std::uint32_t weight_index = rand()%(network->weight_table_size());
      double weight_value = static_cast<double>(rand()%20) / (15.0);
      context->set_network_weight(weight_index, weight_value);
      reference_context.set_network_weight(weight_index, weight_value);
    }/*for(5 consecutive steps)*/
  }/*for(10 variants)*/
}

TEST_CASE("Testing weight updates with the GPU context","[context][GPU][weight-update][bulk]"){
  google::protobuf::Arena arena;
  std::uint32_t sequence_size = rand()%3 + 1;
  std::uint32_t number_of_sequences = rand()%10 + 2;
  std::uint32_t feature_size = rand()%5 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_learning_rate((0.1))
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
    .input_size(2).expected_input_range((1.0))
    .set_recurrence_to_layer()
    .allowed_transfer_functions_by_layer(
      {
        {rafko_net::transfer_function_identity},
        {rafko_net::transfer_function_sigmoid},
        {rafko_net::transfer_function_tanh},
        {rafko_net::transfer_function_elu},
        {rafko_net::transfer_function_selu},
        {rafko_net::transfer_function_relu},
      }
    ).dense_layers({2,2,2,2,2,feature_size});
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> context;
  CHECK_NOTHROW(
    context = (
      rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
        .select_platform().select_device()
        .build()
    )
  );
  rafko_net::RafkoNet network_copy = rafko_net::RafkoNet(*network);
  rafko_mainframe::RafkoCPUContext reference_context(network_copy, settings);

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    settings, rafko_gym::cost_function_cross_entropy
  );
  reference_context.set_objective(objective);
  context->set_objective(objective);

  for(std::uint32_t variant = 0u; variant < 10u; ++variant){
    number_of_sequences = rand()%10 + 1;
    sequence_size = rand()%10 + 1;
    (void)context->expose_settings().set_memory_truncation(sequence_size);
    (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
    std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
      2/* input size */, feature_size,
      number_of_sequences, sequence_size, 2/*prefill_size*/,
      rand()%100/*expected_label*/, (1.0)
    ) );
    std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(*dataset);

    context->set_environment(environment);
    reference_context.set_environment(environment);

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );

      std::vector<double> weight_deltas(network->weight_table_size());
      std::generate(weight_deltas.begin(), weight_deltas.end(), [](){
        return static_cast<double>(rand()%100) / (100.0);
      });
      context->set_network_weights(weight_deltas);
      reference_context.set_network_weights(weight_deltas);
    }/*for(5 consecutive steps)*/

    for(std::uint32_t steps = 0; steps < 5; ++steps){
      REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );

      REQUIRE( context->expose_network().weight_table_size() == reference_context.expose_network().weight_table_size() );
      for(std::int32_t weight_index = 0; weight_index < context->expose_network().weight_table_size(); ++weight_index){
        REQUIRE( context->expose_network().weight_table(weight_index) == reference_context.expose_network().weight_table(weight_index) );
      }

      std::vector<double> weight_deltas(network->weight_table_size());
      std::generate(weight_deltas.begin(), weight_deltas.end(), [](){
        return static_cast<double>(rand()%100) / (100.0);
      });

      context->apply_weight_update(weight_deltas);
      reference_context.apply_weight_update(weight_deltas);
    }/*for(5 consecutive steps)*/
  }/*for(10 variants)*/
}

} /* namespace rako_gym_test */
