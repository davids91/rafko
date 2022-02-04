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
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/training.pb.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_dataset_cost.h"
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

TEST_CASE("Testing if a standalone solution is working as intended with the GPU context","[context][GPU][solve]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 50u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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
    std::vector<sdouble32> network_input(network->input_data_size(), (rand()%10));
    rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
    rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);

    (void)rafko_net::SolutionBuilder::get_kernel_for_solution(*reference_solution, "aw_yiss", sequence_size, 0, settings);
    for(uint32 result_index = 0; result_index < reference_result.size(); ++result_index){
      CHECK( Catch::Approx(reference_result[result_index]).epsilon(0.0000000001) == context_result[result_index] );
    }
  }/*for(50 variants)*/
}

TEST_CASE("Testing if a standalone solution is working as intended with the GPU context even with inputs from the past","[context][GPU][solve][memory]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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
    std::vector<sdouble32> network_input(network->input_data_size(), (rand()%10));
    for(uint32 steps = 0; steps < 5; ++steps){
      rafko_utilities::ConstVectorSubrange<> reference_result = reference_agent->solve(network_input);
      rafko_utilities::ConstVectorSubrange<> context_result = context->solve(network_input);
      for(uint32 result_index = 0; result_index < reference_result.size(); ++result_index){
        CHECK( Catch::Approx(reference_result[result_index]).epsilon(0.0000000001) == context_result[result_index] );
      }
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with single sample of sequence size one","[context][GPU][evaluate]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 1u;
  uint32 number_of_sequences = 1u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoDatasetCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );
    /* TODO: why is it needed to decrease punctuality to 10 digits? why??? */

    for(uint32 steps = 0; steps < 1; ++steps){
      std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<sdouble32>>(std::get<0>(tmp1)),
        std::vector<std::vector<sdouble32>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(uint32 i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels","[context][GPU][evaluate][multi-label]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 1u;
  uint32 number_of_sequences = 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoDatasetCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(uint32 steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 1;
      (void)context->expose_settings().set_memory_truncation(sequence_size);
      (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
      std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<sdouble32>>(std::get<0>(tmp1)),
        std::vector<std::vector<sdouble32>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(uint32 i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels and sequential data","[context][GPU][evaluate][multi-label][sequence]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 1u;
  uint32 number_of_sequences = 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoDatasetCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );

    for(uint32 steps = 0; steps < 5; ++steps){
      number_of_sequences = rand()%10 + 1;
      sequence_size = rand()%10 + 1;
      (void)context->expose_settings().set_memory_truncation(sequence_size);
      (void)reference_context.expose_settings().set_memory_truncation(sequence_size);
      std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> tmp1 = (
        rafko_test::create_sequenced_addition_dataset(number_of_sequences, sequence_size)
      );
      std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
        std::vector<std::vector<sdouble32>>(std::get<0>(tmp1)),
        std::vector<std::vector<sdouble32>>(std::get<1>(tmp1)),
        sequence_size
      );

      context->set_environment(environment);
      reference_context.set_environment(environment);

      for(uint32 i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

TEST_CASE("Testing full evaluation with the GPU context with multiple labels and sequential data and prefill","[context][GPU][evaluate][multi-label][sequence][prefill]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 1u;
  uint32 number_of_sequences = 1;
  uint32 feature_size = 2;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNet* network = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0))
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

    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoDatasetCost>(
      settings, rafko_gym::cost_function_squared_error
    );

    reference_context.set_objective(objective);
    context->set_objective(objective);
    REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.0000000001) == context->full_evaluation() );
    /*!Note: if 15 digits are used for comparison instead of 10 sometimes there's a mismatch */

    for(uint32 steps = 0; steps < 5; ++steps){
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

      for(uint32 i = 0; i < 3; ++i)
        REQUIRE( Catch::Approx(reference_context.full_evaluation()).epsilon(0.00000000000001) == context->full_evaluation() );
    }/*for(5 consecutive steps)*/
  }/*for(10 variant)*/
}

/* TODO: try to test stochastic evaluation.. */
/* TODO: test features as well */
/* TODO: change cost functions in tests*/

} /* namespace rako_gym_test */