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

#include <iomanip>
#include <iostream>
#include <limits>

#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/models/rafko_dataset_implementation.hpp"
#include "rafko_gym/services/cost_function_mse.hpp"
#include "rafko_gym/services/function_factory.hpp"
#include "rafko_gym/services/rafko_numeric_optimizer.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_cpu_context.hpp"
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#if (RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_gpu_context.hpp"
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"
#endif /*(RAFKO_USES_OPENCL)*/

#include "test/test_utility.hpp"

namespace rafko_gym_test {

TEST_CASE("Stress-testing big input takein", "[bigpic][.][!benchmark]") {
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<
      rafko_mainframe::RafkoSettings>(
      rafko_mainframe::RafkoSettings()
          .set_learning_rate(8e-2)
          .set_minibatch_size(64)
          .set_memory_truncation(2)
          .set_droput_probability(0.2)
          .set_training_strategy(
              rafko_gym::Training_strategy::
                  training_strategy_stop_if_training_error_zero,
              true)
          .set_training_strategy(
              rafko_gym::Training_strategy::training_strategy_early_stopping,
              false)
          .set_learning_rate_decay({{1000u, 0.8}})
          .set_arena_ptr(&arena)
          .set_max_solve_threads(2)
          .set_max_processing_threads(4));

  std::shared_ptr<rafko_gym::RafkoObjective> objective =
      std::make_shared<rafko_gym::RafkoCost>(
          *settings, rafko_gym::cost_function_squared_error);
  std::uint32_t avg_ms = 0;
  for (std::uint32_t i = 0; i < 6; ++i) {
    avg_ms = 0;
    std::cout << "image size: " << std::pow(2, (3 + i));
    for (std::uint32_t runs = 0; runs < 10; ++runs) {
      std::uint32_t input_size = std::pow(2, (i)) /*w*/ *
                                 std::pow(2, (i)) /*h*/ * 3 /*rgb*/ *
                                 3 /*pictures*/;
      std::vector<double> input(input_size, 5.0);
      rafko_net::RafkoNet &network =
          *rafko_net::RafkoNetBuilder(*settings)
               .input_size(input_size)
               .expected_input_range(1.0)
               .allowed_transfer_functions_by_layer({
                   {rafko_net::transfer_function_selu},
                   {rafko_net::transfer_function_selu},
                   {rafko_net::transfer_function_selu},
               })
               .create_layers({2, 2, 1});
      std::chrono::steady_clock::time_point start;

      start = std::chrono::steady_clock::now();

#if (RAFKO_USES_OPENCL)
      std::shared_ptr<rafko_mainframe::RafkoContext> context1(
          rafko_mainframe::RafkoOCLFactory()
              .select_platform()
              .select_device()
              .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                       objective));
#else
      std::shared_ptr<rafko_mainframe::RafkoContext> context1 =
          std::make_shared<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                             objective);
#endif /*(RAFKO_USES_OPENCL)*/

      auto current_duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      std::uint32_t average_duration = 0;
      for (std::uint32_t j = 0; j < 500; ++j) {
        start = std::chrono::steady_clock::now();
        (void)context1->solve(input, false);
        auto current_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (0 == average_duration)
          average_duration = current_duration;
        else
          average_duration = (average_duration + current_duration) / 2;
        std::cout << "\rrun duration: " << current_duration
                  << "ms; \t\tavg:" << average_duration << "ms      ";
      }
      std::cout << "-" << std::flush;
      avg_ms += current_duration;
    } /*for(runs)*/
    std::cout << ">" << (avg_ms / 10) << "ms" << std::endl;
  } /*for(sizes)*/
}

/*###############################################################################################
 * Testing if the gradients are added to the fragment correctly
 * */
TEST_CASE("Testing aproximization fragment handling",
          "[numeric_optimization][fragments]") {
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings =
      std::make_shared<rafko_mainframe::RafkoSettings>(
          rafko_mainframe::RafkoSettings()
              .set_max_processing_threads(7)
              .set_learning_rate(1e-1)
              .set_arena_ptr(&arena));

  /* Create nets */
  /*!Note: no need for smart pointers, because ownership is in the arena.
   * The builder automatically uses the arena pointer provided in the settings.
   */
  rafko_net::RafkoNet &network = *rafko_net::RafkoNetBuilder(*settings)
                                      .input_size(2)
                                      .expected_input_range(1.0)
                                      .allowed_transfer_functions_by_layer(
                                          {{rafko_net::transfer_function_selu}})
                                      .create_layers({1});

  std::shared_ptr<rafko_gym::RafkoObjective> objective =
      std::make_shared<rafko_gym::RafkoCost>(
          *settings, rafko_gym::cost_function_squared_error);

  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context =
      std::make_shared<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  rafko_gym::RafkoNumericOptimizer approximizer({context}, {} /*test_context*/,
                                                settings);

  /* adding a simple-weight-gradient fragment */
  std::uint32_t weight_index = rand() % (network.weight_table_size());
  std::uint32_t gradient_value_index;
  double weight_gradient = (0.5);
  double weight_old_value = network.weight_table(weight_index);

  REQUIRE(network.weight_table(weight_index) == weight_old_value);

  approximizer.add_to_fragment(weight_index, weight_gradient);
  CHECK(1 == approximizer.get_fragment().values_size());
  CHECK(1 == approximizer.get_fragment().weight_synapses_size());
  CHECK(weight_gradient == approximizer.get_fragment().values(0));
  gradient_value_index =
      approximizer.get_fragment().weight_synapses(0).starts();
  REQUIRE(static_cast<std::int32_t>(gradient_value_index) <
          network.weight_table_size());

  approximizer.apply_weight_vector_delta(); /* Add the negative gradient */
  REQUIRE(
      (weight_old_value - (weight_gradient * settings->get_learning_rate())) ==
      Catch::Approx(network.weight_table(weight_index))
          .epsilon(0.00000000000001));

  REQUIRE((network.weight_table(weight_index) +
           (weight_gradient * settings->get_learning_rate())) ==
          Catch::Approx(weight_old_value).epsilon(0.00000000000001));

  /* Continously adding gradients into a single fragment, while redundantly
   * collecting them to see that the effect is the same */
  std::vector<double> correct_weight_delta(network.weight_table_size(), (0.0));
  std::vector<double> initial_weights = {network.weight_table().begin(),
                                         network.weight_table().end()};
  for (std::uint32_t variant = 0; variant < 10; ++variant) {
    weight_index = rand() % (network.weight_table_size());
    weight_gradient = (10.0) / static_cast<double>(rand() % 10 + 1);
    correct_weight_delta[weight_index] += weight_gradient;
    approximizer.add_to_fragment(weight_index, weight_gradient);
  }
  for (weight_index = 0;
       static_cast<std::int32_t>(weight_index) < network.weight_table_size();
       ++weight_index) {
    REQUIRE(
        network.weight_table(weight_index) ==
        Catch::Approx(initial_weights[weight_index]).epsilon(0.00000000000001));
  }
  approximizer.apply_weight_vector_delta();
  for (weight_index = 0;
       static_cast<std::int32_t>(weight_index) < network.weight_table_size();
       ++weight_index) {
    CHECK(Catch::Approx(network.weight_table(weight_index))
              .epsilon(0.00000000000001) ==
          (initial_weights[weight_index] - (correct_weight_delta[weight_index] *
                                            settings->get_learning_rate())));
  }
}

TEST_CASE("Testing if numeric optimizer converges networks",
          "[optimize][CPU][small]") {
  return; /*!Note: This testcase is for fallback only, in case the next one does
             not work properly */
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<
      rafko_mainframe::RafkoSettings>(
      rafko_mainframe::RafkoSettings()
          .set_learning_rate(0.5)
          .set_minibatch_size(64)
          .set_memory_truncation(2)
          .set_droput_probability(0.2)
          .set_training_strategy(
              rafko_gym::Training_strategy::
                  training_strategy_stop_if_training_error_zero,
              true)
          .set_training_strategy(
              rafko_gym::Training_strategy::training_strategy_early_stopping,
              false)
          .set_learning_rate_decay({{100u, 0.5},
                                    {200u, 0.3},
                                    {300u, 0.1},
                                    {500u, 0.1},
                                    {1000u, 0.1}})
          .set_arena_ptr(&arena)
          .set_max_solve_threads(2)
          .set_max_processing_threads(4));

  rafko_net::RafkoNet &network =
      *rafko_net::RafkoNetBuilder(*settings)
           .input_size(2)
           .expected_input_range(1.0)
           .add_feature_to_layer(0u,
                                 rafko_net::neuron_group_feature_boltzmann_knot)
           // .add_feature_to_layer(1u, rafko_net::neuron_group_feature_softmax)
           // .add_feature_to_layer(0u,
           // rafko_net::neuron_group_feature_l1_regularization)
           // .add_feature_to_layer(1u,
           // rafko_net::neuron_group_feature_l1_regularization)
           .add_neuron_recurrence(0u /*layer_index*/, 0u /*layer_neuron_index*/,
                                  1u /*past*/)
           .add_neuron_recurrence(0u /*layer_index*/, 1u /*layer_neuron_index*/,
                                  1u /*past*/)
           .add_neuron_recurrence(0u /*layer_index*/, 2u /*layer_neuron_index*/,
                                  1u /*past*/)
           .add_neuron_recurrence(1u /*layer_index*/, 0u /*layer_neuron_index*/,
                                  1u /*past*/)
           .set_neuron_input_function(0u, 0u, rafko_net::input_function_add)
           .set_neuron_input_function(0u, 1u, rafko_net::input_function_add)
           .set_neuron_input_function(0u, 2u, rafko_net::input_function_add)
           .set_neuron_input_function(1u, 0u, rafko_net::input_function_add)
           .allowed_transfer_functions_by_layer(
               {// {rafko_net::transfer_function_selu},
                // {rafko_net::transfer_function_selu},
                // {rafko_net::transfer_function_selu},
                // {rafko_net::transfer_function_selu},
                // {rafko_net::transfer_function_selu},
                {rafko_net::transfer_function_selu},
                {rafko_net::transfer_function_selu}})
           .create_layers({3, 1});
  // .create_layers({30,1});
  // .create_layers({10,15,20,15,10,5,1});

  // network->mutable_weight_table()->Set(22,0.777);

  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment =
      std::make_shared<rafko_gym::RafkoDatasetImplementation>(
          std::vector<std::vector<double>>{// {0.777, 0.777},{0.777, 0.777},
                                           {0.666, 0.666},
                                           {0.666, 0.666}},
          std::vector<std::vector<double>>{// {11.0},{21.0},
                                           {10.0},
                                           {20.0}},
          2 /*sequence_size*/
      );

  std::shared_ptr<rafko_gym::RafkoObjective> objective =
      std::make_shared<rafko_gym::RafkoCost>(
          *settings, rafko_gym::cost_function_squared_error);
#if (RAFKO_USES_OPENCL)
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> context1(
      rafko_mainframe::RafkoOCLFactory()
          .select_platform()
          .select_device()
          .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                   objective));
  settings->set_max_processing_threads(1u);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context2 =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> test_context(
      rafko_mainframe::RafkoOCLFactory()
          .select_platform()
          .select_device()
          .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                   objective));
  rafko_gym::RafkoNumericOptimizer approximizer({context1, context2},
                                                {} /*test_context*/, settings);
  context2->set_data_set(environment);
  context2->set_weight_updater(rafko_gym::weight_updater_amsgrad);
  context2->set_objective(objective);
#else
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context1 =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> test_context =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  rafko_gym::RafkoNumericOptimizer approximizer({context1}, {} /*test_context*/,
                                                settings);
#endif /*(RAFKO_USES_OPENCL)*/
  context1->set_data_set(environment);
  context1->set_weight_updater(rafko_gym::weight_updater_amsgrad);
  context1->set_objective(objective);
  test_context->set_objective(objective);
  std::vector<std::vector<double>> actual_value(2, std::vector<double>(2, 0.0));
  std::uint32_t iteration = 0u;
  std::uint32_t avg_duration = 0.0;
  std::chrono::steady_clock::time_point start;
  rafko_net::SolutionSolver::Factory reference_solver_factory(network,
                                                              settings);
  while ((std::abs(actual_value[1][0] - environment->get_label_sample(0u)[0]) +
          std::abs(actual_value[0][0] - environment->get_label_sample(1u)[0])) >
         0.002) {
    start = std::chrono::steady_clock::now();
    approximizer.collect_approximates_from_weight_gradients();
    auto current_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (0.0 == avg_duration)
      avg_duration = current_duration;
    else
      avg_duration = (avg_duration + current_duration) / 2.0;
    approximizer.apply_weight_vector_delta();
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver =
        reference_solver_factory.build();
    actual_value[1][0] =
        reference_solver->solve(environment->get_input_sample(0u), true, 0u)[0];
    actual_value[0][0] = reference_solver->solve(
        environment->get_input_sample(1u), false, 0u)[0];

    double weight_sum = std::accumulate(
        network.weight_table().begin(), network.weight_table().end(), 0.0,
        [](const double &accu, const double &element) {
          return accu + std::abs(element);
        });
    std::cout << "Target: " << environment->get_label_sample(0u)[0]
              << " --?--> " << actual_value[1][0] << ";   "
              << environment->get_label_sample(1u)[0] << " --?--> "
              << actual_value[0][0] << " | avg duration: " << avg_duration
              << "ms "
              << " | weight_sum: " << weight_sum
              << " | iteration: " << iteration << "     \r";
    ++iteration;
  }
  std::cout << "\nTarget reached in " << iteration << " iterations!    "
            << std::endl;
}

TEST_CASE("Testing basic aproximization",
          "[numeric_optimization][feed-forward][.][!benchmark]") {
  google::protobuf::Arena arena;
#if (RAFKO_USES_OPENCL)
  std::uint32_t number_of_samples = 1024;
  std::uint32_t minibatch_size = 256;
#else
  std::uint32_t number_of_samples = 128;
  std::uint32_t minibatch_size = 32;
#endif /*(RAFKO_USES_OPENCL)*/

  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<
      rafko_mainframe::RafkoSettings>(
      rafko_mainframe::RafkoSettings()
          .set_learning_rate(8e-1)
          .set_minibatch_size(minibatch_size)
          .set_memory_truncation(2)
          .set_droput_probability(0.2)
          .set_training_strategy(
              rafko_gym::Training_strategy::
                  training_strategy_stop_if_training_error_zero,
              true)
          .set_training_strategy(
              rafko_gym::Training_strategy::training_strategy_early_stopping,
              false)
          .set_learning_rate_decay({{1000u, 0.8}})
          .set_arena_ptr(&arena)
          .set_max_solve_threads(2)
          .set_max_processing_threads(4));

  /* Create network */
  rafko_net::RafkoNet &network =
      *rafko_net::RafkoNetBuilder(*settings)
           .input_size(2)
           .expected_input_range(1.0)
           .add_feature_to_layer(1,
                                 rafko_net::neuron_group_feature_boltzmann_knot)
           // .add_feature_to_layer(0,
           // rafko_net::neuron_group_feature_l1_regularization)
           // .add_feature_to_layer(0,
           // rafko_net::neuron_group_feature_l2_regularization)
           // .add_feature_to_layer(1,
           // rafko_net::neuron_group_feature_l1_regularization)
           .add_feature_to_layer(
               1, rafko_net::neuron_group_feature_l2_regularization)
           // // .add_feature_to_layer(2,
           // rafko_net::neuron_group_feature_l1_regularization)
           .add_feature_to_layer(
               2, rafko_net::neuron_group_feature_l2_regularization)
           // .add_feature_to_layer(1,
           // rafko_net::neuron_group_feature_dropout_regularization)
           .allowed_transfer_functions_by_layer({
               {rafko_net::transfer_function_selu},
               {rafko_net::transfer_function_selu},
               {rafko_net::transfer_function_selu},
           })
           .create_layers({2, 2, 1});

  /* Create dataset, test set and optimizers; optimize nets */
  std::shared_ptr<rafko_gym::RafkoObjective> objective =
      std::make_shared<rafko_gym::RafkoCost>(
          *settings, rafko_gym::cost_function_squared_error);
  settings->set_max_processing_threads(1u);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context2 =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  auto [inputs1, labels1] = rafko_test::create_sequenced_addition_dataset(
      number_of_samples, /*sequence_size*/ 4);
  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment =
      std::make_shared<rafko_gym::RafkoDatasetImplementation>(
          std::move(inputs1), std::move(labels1), /* Sequence size */ 4);

#if (RAFKO_USES_OPENCL)
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> context1(
      rafko_mainframe::RafkoOCLFactory()
          .select_platform()
          .select_device()
          .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                   objective));
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> test_context(
      rafko_mainframe::RafkoOCLFactory()
          .select_platform()
          .select_device()
          .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                   objective));
  rafko_gym::RafkoNumericOptimizer approximizer({context1, context2},
                                                {} /*test_context*/, settings);
  context2->set_data_set(environment);
  context2->set_weight_updater(rafko_gym::weight_updater_amsgrad);
#else
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context1 =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> test_context =
      std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  rafko_gym::RafkoNumericOptimizer approximizer({context1}, {} /*test_context*/,
                                                settings);
#endif /*(RAFKO_USES_OPENCL)*/

  approximizer.set_weight_filter(1.0);
  context1->set_data_set(environment);
  context1->set_weight_updater(rafko_gym::weight_updater_amsgrad);

  auto [inputs2, labels2] =
      rafko_test::create_sequenced_addition_dataset(number_of_samples, 4);
  test_context->set_data_set(
      std::make_shared<rafko_gym::RafkoDatasetImplementation>(
          std::move(inputs2), std::move(labels2), /* Sequence size */ 4));

  auto [inputs3, labels3] =
      rafko_test::create_sequenced_addition_dataset(number_of_samples * 2, 4);
  rafko_gym::RafkoDatasetImplementation *after_test_set =
      google::protobuf::Arena::Create<rafko_gym::RafkoDatasetImplementation>(
          settings->get_arena_ptr(), std::move(inputs3), std::move(labels3),
          /* Sequence size */ 4);
  settings->get_arena_ptr()->Own(after_test_set);

  double train_error = 1.0;
  double test_error = 1.0;
  double minimum_error;
  double low_error = 0.025;
  std::uint32_t iteration_reached_low_error =
      std::numeric_limits<std::uint32_t>::max();
  std::uint32_t iteration;
  std::chrono::steady_clock::time_point start;
  std::uint32_t average_duration;
  double avg_gradient;

  train_error = 1.0;
  test_error = 1.0;
  average_duration = 0;
  iteration = 0;
  minimum_error = std::numeric_limits<double>::max();

  std::cout << "Approximizing network:" << std::endl;
  std::cout << "Training Error; \t\tTesting Error; min; \t\t avg_d_w_abs; \t\t "
               "iteration; \t\t duration(ms); avg duration(ms)\t "
            << std::endl;
  std::cout.precision(15);
  while (!approximizer.stop_training()) {
    start = std::chrono::steady_clock::now();
    approximizer.collect_approximates_from_weight_gradients();
    avg_gradient = 0;
    for (std::int32_t frag_index = 0;
         frag_index < approximizer.get_weight_gradient().values_size();
         ++frag_index) {
      avg_gradient +=
          std::abs(approximizer.get_weight_gradient().values(frag_index));
    }
    avg_gradient /= std::max(
        std::numeric_limits<double>::epsilon(),
        static_cast<double>(approximizer.get_weight_gradient().values_size()));

    approximizer.apply_weight_vector_delta();
    auto current_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    average_duration += current_duration;
    train_error = approximizer.get_error_estimation();
    test_context->refresh_solution_weights();
    test_error = -test_context->full_evaluation();
    if (abs(test_error) < minimum_error) {
      minimum_error = abs(test_error);
      std::cout << std::endl;
    }
    std::cout << "\r";
    for (std::uint32_t space_count = 0;
         space_count < rafko_test::get_console_width() - 1; ++space_count)
      std::cout << " ";
    std::cout << "\r";
    std::cout << std::setprecision(9) << train_error << ";\t\t" << test_error
              << "; " << minimum_error << ";\t\t" << avg_gradient << ";\t\t"
              << iteration << ";\t\t" << current_duration << "; "
              << average_duration / static_cast<double>(iteration + 1)
              << ";\t\t"
              << "\r" << std::flush;
    if (0 == (iteration % 100)) {
      srand(iteration);
      approximizer.full_evaluation();
    }
    ++iteration;
    // if(250 == iteration){
    //   approximizer.set_weight_filter({
    //     1.0, 0.0, 0.0, 0.0, 0.0,  /* Neuron 0 */
    //     1.0, 0.0, 0.0, 0.0, 0.0,  /* Neuron 1 */
    //     1.0, 0.0, 0.0, 0.0, 0.0,  /* Neuron 2 */
    //     1.0, 0.0, 0.0, 0.0, 0.0,  /* Neuron 3 */
    //     1.0, 0.0, 0.0, 0.0, 0.0  /* Neuron 4 */
    //   });
    // }
    approximizer.set_weight_exclude_chance_filter(
        static_cast<double>(std::min(800u, iteration)) / 1000.0);
    if (test_error <= low_error) {
      iteration_reached_low_error =
          std::min(iteration_reached_low_error, iteration);
      if ((iteration - iteration_reached_low_error) > 200) {
        break; /* End the loop if 200 loops spent below error threshold */
      }
    }
  }
  average_duration /= static_cast<double>(iteration + 1);
  std::cout << std::endl
            << "Optimum reached in " << (iteration + 1)
            << " steps!(average runtime: " << average_duration << " ms)"
            << std::endl;

  double error_summary[3] = {0, 0, 0};
  rafko_gym::CostFunctionMSE after_cost(*settings);
  for (std::uint32_t i = 0; i < number_of_samples; ++i) {
    bool reset = 0 == (i % (after_test_set->get_sequence_size()));
    rafko_utilities::ConstVectorSubrange<> neuron_data =
        test_context->solve(after_test_set->get_input_sample(i), reset);
    error_summary[0] += after_cost.get_feature_error(
        {neuron_data.begin(), neuron_data.end()},
        after_test_set->get_label_sample(i), number_of_samples);
  }
  std::cout << "==================================\n Error summaries:"
            << "\t" << error_summary[0] << std::endl;
}

} /* namespace rafko_gym_test */
