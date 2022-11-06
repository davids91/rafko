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
#include <iostream>
#include <iomanip>

#ifdef WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_cpu_context.hpp"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"
#include "rafko_mainframe/services/rafko_gpu_context.hpp"
#include "rafko_gym/services/rafko_autodiff_gpu_optimizer.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_net/services/solution_solver.hpp"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/models/rafko_dataset_implementation.hpp"
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

#include "test/test_utility.hpp"

namespace rafko_gym_test {

TEST_CASE("Testing if autodiff optimizer converges networks", "[optimize][small][manual]"){
  return; /*!Note: This testcase is for fallback only, in case the next one does not work properly */
  double learning_rate = 0.0001;
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    .set_learning_rate(8e-2).set_minibatch_size(64).set_memory_truncation(1)
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4)
  );
  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(2).expected_input_range(1.0)
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_boltzmann_knot)
    .add_neuron_recurrence(1u,0u,1u)
    .set_neuron_input_function(0u, 0u, rafko_net::input_function_multiply)
    // .set_neuron_spike_function(0u, 0u, rafko_net::spike_function_none)
    // .set_neuron_input_function(0u, 1u, rafko_net::input_function_add)
    // .set_neuron_spike_function(0u, 1u, rafko_net::spike_function_none)
    // .set_neuron_input_function(0u, 2u, rafko_net::input_function_add)
    // .set_neuron_spike_function(0u, 2u, rafko_net::spike_function_none)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .create_layers({3,1});

  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment = std::make_shared<rafko_gym::RafkoDatasetImplementation>(
    std::vector<std::vector<double>>{{1.0,1.0},{1.0,1.0}},
    std::vector<std::vector<double>>{{1.0},{2.0}},
    2.0/*sequence_size*/
  );

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_squared_error
  );

  rafko_gym::RafkoAutodiffOptimizer optimizer(settings, environment, network);
  optimizer.build(objective);
  std::vector<std::vector<double>> actual_value(2, std::vector<double>(2, 0.0));
  std::uint32_t iteration = 0u;
  rafko_net::SolutionSolver::Factory reference_solver_factory(network, settings);
  while(
    (
      std::abs(actual_value[1][0] - environment->get_label_sample(0u)[0])
      + std::abs(actual_value[0][0] - environment->get_label_sample(1u)[0])
    ) > (2.0 * learning_rate)
  ){
    optimizer.reset();
    optimizer.calculate(
      {environment->get_input_samples().begin(), environment->get_input_samples().end()},
      {environment->get_label_samples().begin(), environment->get_label_samples().end()}
    );
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver = reference_solver_factory.build();
    for(std::int32_t weight_index = 0; weight_index < network.weight_table_size(); ++weight_index){
      network.set_weight_table(
        weight_index,
        ( network.weight_table(weight_index) - (optimizer.get_avg_gradient(weight_index) * learning_rate) )
      );
    }
    actual_value[1][0] = optimizer.get_neuron_operation(3u)->get_value(1u/*past_index*/);
    actual_value[0][0] = optimizer.get_neuron_operation(3u)->get_value(0u/*past_index*/);
    REQUIRE(
      reference_solver->solve(environment->get_input_sample(0u), true, 0u)[0]
      == Catch::Approx(actual_value[1][0]).epsilon(0.0000000000001)
    );
    REQUIRE(
      reference_solver->solve(environment->get_input_sample(1u), false, 0u)[0]
      == Catch::Approx(actual_value[0][0]).epsilon(0.0000000000001)
    );
    std::cout << "Target: "
    << environment->get_label_sample(0u)[0] << " --?--> " << actual_value[1][0] << ";   "
    << environment->get_label_sample(1u)[0] << " --?--> " << actual_value[0][0]
    << "     \r";
    ++iteration;
  }
  std::cout << "\nTarget reached in " << iteration << " iterations!    " << std::endl;
}

TEST_CASE("Testing if autodiff optimizer converges networks with the iteration interface", "[optimize][small]"){
  return; /*!Note: This testcase is for fallback only, in case the next one does not work properly */
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    .set_learning_rate(0.0001).set_minibatch_size(64).set_memory_truncation(2)
    .set_droput_probability(0.2)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero,true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping,false)
    .set_learning_rate_decay({{1000u,0.8}})
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4)
  );

  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(2).expected_input_range(1.0)
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_boltzmann_knot)
    .set_neuron_input_function(0u, 0u, rafko_net::input_function_multiply)
    .set_neuron_spike_function(1u, 0u, rafko_net::spike_function_p)
    .set_neuron_spike_function(0u, 1u, rafko_net::spike_function_memory)
    .set_neuron_spike_function(0u, 2u, rafko_net::spike_function_memory)
    .set_neuron_spike_function(0u, 3u, rafko_net::spike_function_memory)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .create_layers({3,1});

  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment = std::make_shared<rafko_gym::RafkoDatasetImplementation>(
    std::vector<std::vector<double>>{{0.666, 0.666},{0.666, 0.666}},
    std::vector<std::vector<double>>{{10.0},{20.0}},
    2 /*sequence_size*/
  );

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_squared_error
  );

  rafko_gym::RafkoAutodiffOptimizer optimizer(settings, environment, network);
  optimizer.build(objective);
  optimizer.set_weight_updater(rafko_gym::weight_updater_default);
  std::vector<std::vector<double>> actual_value(2, std::vector<double>(2, 0.0));
  std::uint32_t iteration = 0u;
  std::uint32_t avg_duration = 0.0;
  std::chrono::steady_clock::time_point start;
  rafko_net::SolutionSolver::Factory reference_solver_factory(network, settings);
  while(
    (
      std::abs(actual_value[1][0] - environment->get_label_sample(0u)[0])
      + std::abs(actual_value[0][0] - environment->get_label_sample(1u)[0])
    ) > (2.0 * settings->get_learning_rate())
  ){
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver = reference_solver_factory.build();
    start = std::chrono::steady_clock::now();
    optimizer.iterate();
    auto current_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if(0.0 == avg_duration)avg_duration = current_duration;
    else avg_duration = (avg_duration + current_duration)/2.0;

    actual_value[1][0] = optimizer.get_neuron_operation(3u)->get_value(1u/*past_index*/);
    actual_value[0][0] = optimizer.get_neuron_operation(3u)->get_value(0u/*past_index*/);
    CHECK(
      reference_solver->solve(environment->get_input_sample(0u), true/*reset_memory*/, 0u/*thread_index*/)[0]
      == Catch::Approx(actual_value[1][0]).epsilon(0.0000000001)
    );
    REQUIRE(
      reference_solver->solve(environment->get_input_sample(1u), false/*reset_memory*/, 0u/*thread_index*/)[0]
      == Catch::Approx(actual_value[0][0]).epsilon(0.0000000001)
    );

    double weight_sum = std::accumulate(
      network.weight_table().begin(), network.weight_table().end(), 0.0,
      [](const double& accu, const double& element){ return accu + std::abs(element); }
    );
    std::cout << "Target: "
    << environment->get_label_sample(0u)[0] << " --?--> " << actual_value[1][0] << ";   "
    << environment->get_label_sample(1u)[0] << " --?--> " << actual_value[0][0]
    << " | avg duration: " << avg_duration << "ms "
    << " | weight_sum: " << weight_sum
    << " | iteration: " << iteration
    << "     \r";
    ++iteration;
  }
  std::cout << "\nTarget reached in " << iteration << " iterations!    " << std::endl;
}

#if(RAFKO_USES_OPENCL)
TEST_CASE("Testing if autodiff GPU optimizer converges networks with the GPU optimizer", "[optimize][GPU][small]"){
  return; /*!Note: This testcase is for fallback only, in case the next one does not work properly */
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    .set_learning_rate(0.0001).set_minibatch_size(64).set_memory_truncation(2)
    .set_droput_probability(0.2)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero,true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping,false)
    .set_learning_rate_decay({{1000u,0.8}})
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4)
  );

  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(2).expected_input_range(1.0)
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_boltzmann_knot)
    // .add_feature_to_layer(1u, rafko_net::neuron_group_feature_softmax)
    // .add_feature_to_layer(0u, rafko_net::neuron_group_feature_l1_regularization)
    // .add_feature_to_layer(1u, rafko_net::neuron_group_feature_l1_regularization)
    .add_neuron_recurrence(0u/*layer_index*/, 0u/*layer_neuron_index*/, 1u/*past*/)
    .add_neuron_recurrence(0u/*layer_index*/, 1u/*layer_neuron_index*/, 1u/*past*/)
    .add_neuron_recurrence(0u/*layer_index*/, 2u/*layer_neuron_index*/, 1u/*past*/)
    .add_neuron_recurrence(1u/*layer_index*/, 0u/*layer_neuron_index*/, 1u/*past*/)
    .set_neuron_input_function(0u, 0u, rafko_net::input_function_add)
    .set_neuron_input_function(0u, 1u, rafko_net::input_function_add)
    .set_neuron_input_function(0u, 2u, rafko_net::input_function_add)
    .set_neuron_input_function(1u, 0u, rafko_net::input_function_add)
    .allowed_transfer_functions_by_layer({
      // {rafko_net::transfer_function_selu},
      // {rafko_net::transfer_function_selu},
      // {rafko_net::transfer_function_selu},
      // {rafko_net::transfer_function_selu},
      // {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .create_layers({3,1});
    // .create_layers({30,1});
    // .create_layers({10,15,20,15,10,5,1});

  // network.mutable_weight_table()->Set(22,0.777);

  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment = std::make_shared<rafko_gym::RafkoDatasetImplementation>(
    std::vector<std::vector<double>>{
      {0.666, 0.666},{0.666, 0.666}
    },
    std::vector<std::vector<double>>{
      {10.0},{20.0}
    },
    2 /*sequence_size*/
  );

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_squared_error
  );

  std::unique_ptr<rafko_gym::RafkoAutodiffGPUOptimizer> optimizerGPU = (
    rafko_mainframe::RafkoOCLFactory()
      .select_platform().select_device()
      .build<rafko_gym::RafkoAutodiffGPUOptimizer>(settings, environment, network)
  );
  optimizerGPU->build(objective);
  optimizerGPU->set_weight_updater(rafko_gym::weight_updater_amsgrad);
  std::vector<std::vector<double>> actual_value(2, std::vector<double>(2, 0.0));
  std::uint32_t iteration = 0u;
  std::uint32_t avg_duration = 0.0;
  std::chrono::steady_clock::time_point start;
  rafko_net::SolutionSolver::Factory reference_solver_factory(network, settings);
  while(
    (
      std::abs(actual_value[1][0] - environment->get_label_sample(0u)[0])
      + std::abs(actual_value[0][0] - environment->get_label_sample(1u)[0])
    ) > (2.0 * settings->get_learning_rate())
  ){
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver = reference_solver_factory.build();
    start = std::chrono::steady_clock::now();
    optimizerGPU->iterate();
    auto current_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if(0.0 == avg_duration)avg_duration = current_duration;
    else avg_duration = (avg_duration + current_duration)/2.0;

    actual_value[1][0] = optimizerGPU->get_neuron_data(0u/*sequence_index*/, 1u/*past_index*/, 3u/*neuron_index*/);
    actual_value[0][0] = optimizerGPU->get_neuron_data(0u/*sequence_index*/, 0u/*past_index*/, 3u/*neuron_index*/);
    CHECK(
      reference_solver->solve(environment->get_input_sample(0u), true, 0u)[0]
      == Catch::Approx(actual_value[1][0]).epsilon(0.0000000001)
    );
    REQUIRE(
      reference_solver->solve(environment->get_input_sample(1u), false, 0u)[0]
      == Catch::Approx(actual_value[0][0]).epsilon(0.0000000001)
    );

    double weight_sum = std::accumulate(
      network.weight_table().begin(), network.weight_table().end(), 0.0,
      [](const double& accu, const double& element){ return accu + std::abs(element); }
    );
    std::cout << "Target: "
    << environment->get_label_sample(0u)[0] << " --?--> " << actual_value[1][0] << ";   "
    << environment->get_label_sample(1u)[0] << " --?--> " << actual_value[0][0]
    << " | avg duration: " << avg_duration << "ms "
    << " | weight_sum: " << weight_sum
    << " | iteration: " << iteration
    << "     \r";
    ++iteration;
  }
  std::cout << "\nTarget reached in " << iteration << " iterations!    " << std::endl;
}
#endif/*(RAFKO_USES_OPENCL)*/

TEST_CASE("Testing if autodiff optimizer converges networks with a prepared environment", "[optimize][!benchmark]"){
  #if(RAFKO_USES_OPENCL)
  std::uint32_t number_of_samples = 1024;
  std::uint32_t minibatch_size = 256;
  #else
  std::uint32_t number_of_samples = 64;
  std::uint32_t minibatch_size = 32;
  #endif/*(RAFKO_USES_OPENCL)*/
  std::uint32_t sequence_size = 4;
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    .set_learning_rate(2e-2).set_minibatch_size(minibatch_size).set_memory_truncation(2)
    .set_droput_probability(0.0)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero, true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping, false)
    .set_learning_rate_decay({{100u,0.8}})
    .set_tolerance_loop_value(10)
    .set_arena_ptr(&arena).set_max_solve_threads(2).set_max_processing_threads(4)
  );

  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(2).expected_input_range(1.0)
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_boltzmann_knot)
    .add_feature_to_layer(1u, rafko_net::neuron_group_feature_boltzmann_knot)
    // .add_feature_to_layer(0u, rafko_net::neuron_group_feature_l1_regularization)
    // .add_feature_to_layer(0u, rafko_net::neuron_group_feature_l2_regularization)
    // .add_neuron_recurrence(1u,0u,1u)
    // .set_neuron_input_function(0u, 0u, rafko_net::input_function_multiply)
    // .set_neuron_spike_function(0u, 0u, rafko_net::spike_function_none)
    // .set_neuron_input_function(0u, 1u, rafko_net::input_function_add)
    // .set_neuron_spike_function(0u, 1u, rafko_net::spike_function_none)
    // .set_neuron_input_function(0u, 2u, rafko_net::input_function_add)
    // .set_neuron_spike_function(0u, 2u, rafko_net::spike_function_none)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_swish},
      {rafko_net::transfer_function_swish},
      {rafko_net::transfer_function_swish}
    })
    .create_layers({3,2,1});

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_squared_error
  );
  #if(RAFKO_USES_OPENCL)
  rafko_mainframe::RafkoOCLFactory factory;
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> context(
    factory.select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(network, settings, objective)
  );
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> test_context(
    factory.select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(network, settings, objective)
  );
  #else
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> context = std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings, objective);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> test_context = std::make_unique<rafko_mainframe::RafkoCPUContext>(network, settings, objective);
  #endif/*(RAFKO_USES_OPENCL)*/
  auto [inputs, labels] = rafko_test::create_sequenced_addition_dataset(number_of_samples, sequence_size);
  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment = std::make_shared<rafko_gym::RafkoDatasetImplementation>(
    std::move(inputs), std::move(labels), sequence_size
  );

  auto [inputs2, labels2] = rafko_test::create_sequenced_addition_dataset(number_of_samples, sequence_size);
  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> test_environment = std::make_shared<rafko_gym::RafkoDatasetImplementation>(
    std::move(inputs2), std::move(labels2), sequence_size
  );

  test_context->set_environment(test_environment);

  #if(RAFKO_USES_OPENCL)
  std::unique_ptr<rafko_gym::RafkoAutodiffGPUOptimizer> optimizer = (
    rafko_mainframe::RafkoOCLFactory()
      .select_platform().select_device()
      .build<rafko_gym::RafkoAutodiffGPUOptimizer>(settings, environment, network, context, test_context)
  );
  #else
  std::unique_ptr<rafko_gym::RafkoAutodiffOptimizer> optimizer = std::make_unique<rafko_gym::RafkoAutodiffOptimizer>(
    settings, environment, network, context, test_context
  );
  #endif/*(RAFKO_USES_OPENCL)*/
  optimizer->build(objective);
  optimizer->set_weight_updater(rafko_gym::weight_updater_amsgrad);
  std::vector<std::vector<double>> actual_value(2, std::vector<double>(2, 0.0));
  double train_error;
  double test_error;
  double minimum_error;
  double low_error = 0.025;
  std::uint32_t iteration_reached_low_error = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t iteration;
  std::chrono::steady_clock::time_point start;
  std::uint32_t avg_duration;

  train_error = 1.0;
  test_error = 1.0;
  avg_duration = 0;
  iteration = 0;
  minimum_error = std::numeric_limits<double>::max();
  std::uint32_t console_width;
  #ifdef WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    int columns, rows;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    console_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
  #else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    console_width = w.ws_col;
  #endif
  rafko_net::SolutionSolver::Factory reference_solver_factory(network, settings);
  std::cout << "Optimizing network:" << std::endl;
  std::cout << "Training Error; \t\tTesting Error; min; \t\t avg_d_w_abs; \t\t iteration; \t\t duration(ms); avg duration(ms)\t " << std::endl;
  while(!optimizer->stop_triggered()){
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver = reference_solver_factory.build();
    start = std::chrono::steady_clock::now();
    optimizer->iterate();
    auto current_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    if(0.0 == avg_duration)avg_duration = current_duration;
    else avg_duration = (avg_duration + current_duration)/2.0;

    train_error = optimizer->get_last_training_error();
    test_error = optimizer->get_last_testing_error();
    if(abs(test_error) < minimum_error){
      minimum_error = abs(test_error);
      std::cout << std::endl;
    }

    std::cout << "\r";
    for(std::uint32_t space_count = 0; space_count < console_width - 1; ++space_count)
      std::cout << " ";
    std::cout << "\r";
    std::cout << std::setprecision(9)
    << train_error << ";\t\t"
    << test_error << "; "
    << minimum_error <<";\t\t"
    << optimizer->get_avg_of_abs_gradient() << ";\t\t"
    << iteration <<";\t\t"
    << current_duration <<"; "
    << avg_duration <<"; "
    << std::flush;
    ++iteration;
    if(std::abs(test_error) <= low_error){
      iteration_reached_low_error = std::min(iteration_reached_low_error, iteration);
      if((iteration - iteration_reached_low_error) > 200){
        std::cout << std::endl << "== good enough for a test ==" << std::endl;
        break; /* End the loop if 200 loops spent below error threshold */
      }
    }
  }
  std::cout << std::endl << "Optimum reached in " << (iteration + 1)
  << " steps!(average runtime: "<< avg_duration << " ms)   " << std::endl;
}

} /* namespace rafko_gym_test */
