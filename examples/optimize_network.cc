#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#ifdef WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <rafko.hpp>
#include <rafko_protocol/rafko_net.pb.h>

int main() {
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<
      rafko_mainframe::RafkoSettings>(
      rafko_mainframe::RafkoSettings()
          .set_max_solve_threads(2)
          .set_max_processing_threads(4)

          /* +++ Standard Training settings +++ */
          .set_learning_rate(2e-7)
          .set_learning_rate_decay(
              {{100u, 0.8},
               {500u,
                0.5}}) /* multiplier applied at the given training iterations */
          .set_minibatch_size(64)
          .set_memory_truncation(
              2) /* limit gradient calculation in case of sequential data */
          .set_droput_probability(0.0)
          .set_training_strategy(
              rafko_gym::Training_strategy::
                  training_strategy_stop_if_training_error_zero,
              true)
          .set_training_strategy(
              rafko_gym::Training_strategy::training_strategy_early_stopping,
              false)
          .set_training_relevant_loop_count(10)
          .set_delta(0.1) /* for early stopping */
  );

  rafko_net::RafkoNet &network =
      *rafko_net::RafkoNetBuilder(*settings)
           .input_size(2)
           .expected_input_range(1.0)
           .add_feature_to_layer(0u,
                                 rafko_net::neuron_group_feature_boltzmann_knot)
           .add_feature_to_layer(1u,
                                 rafko_net::neuron_group_feature_boltzmann_knot)
           .allowed_transfer_functions_by_layer(
               {{rafko_net::transfer_function_selu,
                 rafko_net::transfer_function_relu},
                {rafko_net::transfer_function_selu},
                {rafko_net::transfer_function_selu}})
           .create_layers({3, 2, 1});

  /* +++ Generating sample environment Data set +++ */
  constexpr const std::uint32_t sequence_size = 5u;
  std::vector<std::vector<double>> environment_inputs;
  std::vector<std::vector<double>> environment_labels;
  for (std::uint32_t sequence_index = 0;
       sequence_index < settings->get_minibatch_size() * 3; ++sequence_index) {
    /*!Note: For this example, since the values don't matter full zero content
     * is fine */
    for (std::uint32_t datapoint_index = 0; datapoint_index < sequence_size;
         ++datapoint_index) {
      environment_inputs.push_back(
          std::vector<double>(network.input_data_size(), rand() % 10));
      environment_labels.push_back(
          std::vector<double>(network.output_neuron_number(), rand() % 10));
      /*!Note: It is possible to add more inputs, than the networks input size
       * here and still have a valid environment Inside each sequence, the
       * number of labels are compared to the size of the sequence, and surplus
       * inputs at the beginning of each sequence are counted as "prefill".
       * Their only purpose is to initialize the networks data buffers for each
       * sequence, np error scores or gradients are calculated from them
       */
    }
  }
  std::shared_ptr<rafko_gym::RafkoDatasetImplementation> environment =
      std::make_shared<rafko_gym::RafkoDatasetImplementation>(
          std::move(environment_inputs), std::move(environment_labels),
          sequence_size);
/* --- Generating sample environment Data set --- */

/*!Note: To optimize a network to a data set, the optimizers can be used as
 * follows */
#if (!USE_OPENCL)
  std::unique_ptr<rafko_gym::RafkoAutodiffOptimizer> optimizer =
      std::make_unique<rafko_gym::RafkoAutodiffOptimizer>(settings, environment,
                                                          network);
#else  /* If OpenCL is supported, the below factory can be used */
  std::unique_ptr<rafko_gym::RafkoAutodiffGPUOptimizer> optimizer =
      (rafko_mainframe::RafkoOCLFactory()
           .select_platform()
           .select_device()
           .build<rafko_gym::RafkoAutodiffGPUOptimizer>(settings, environment,
                                                        network));
#endif /*(USE_OPENCL)*/

  /*!Note: In order to construct the backpropagation structure of the network
   * the @rafko_gym::RafkoAutodiffOptimizer::build need be called. It needs an
   * objective to finalise the math formula. This step is required before the
   * training iterations can take place.
   */
  std::shared_ptr<rafko_gym::RafkoObjective> objective =
      std::make_shared<rafko_gym::RafkoCost>(
          *settings, rafko_gym::cost_function_squared_error);
  optimizer->build(objective);

/*!Note: Optimizer works based on gradients, but to see the progress of
 * training, and to enable more advanced features such as
 * rafko_gym::Training_strategy::training_strategy_early_stopping, @RafkoCOntext
 * objects need to be set either in the constructor of the optimizer, or (in
 * this example) set explicitly by modifiers.
 */
#if (!USE_OPENCL)
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> training_context =
      std::make_shared<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
  std::shared_ptr<rafko_mainframe::RafkoCPUContext> test_context =
      std::make_shared<rafko_mainframe::RafkoCPUContext>(network, settings,
                                                         objective);
#else  /* If OpenCL is supported, the below factory can be used */
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> training_context =
      (rafko_mainframe::RafkoOCLFactory()
           .select_platform()
           .select_device()
           .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                    objective));
  std::shared_ptr<rafko_mainframe::RafkoGPUContext> test_context =
      (rafko_mainframe::RafkoOCLFactory()
           .select_platform()
           .select_device()
           .build<rafko_mainframe::RafkoGPUContext>(network, settings,
                                                    objective));
#endif /*(USE_OPENCL)*/

  optimizer->set_training_context(training_context);
  optimizer->set_testing_context(test_context);

  /*!Note: Training can use more advanced features, such as weight updaters. */
  optimizer->set_weight_updater(rafko_gym::weight_updater_momentum);

  /* Initial values to start a training session */
  double train_error = 1.0;
  double test_error = 1.0;
  double minimum_error = std::numeric_limits<double>::max();

  /*!Note: in this example a "good enough" result will be achieved if
   * the error rate is below @low_error for at least @iteration_optimum_delta
   * iterations
   */
  constexpr const double low_error = 0.025;
  constexpr const std::uint32_t iteration_optimum_delta = 200;
  std::uint32_t iteration_reached_low_error =
      std::numeric_limits<std::uint32_t>::max();
  std::uint32_t iteration = 0;
  std::chrono::steady_clock::time_point
      start; /* To display runtime statistics, the optimization is going to be
                timed */
  std::uint32_t avg_duration;

  /*!Note: Calcualte the width of the used console, this helps clear the given
   * row, so data from one iteration can overwrite the current row unless it is
   * a new error minimum. Clearing the row beforehand makes sure that the
   * numbers contain no thrash data from a previous iteration (displayed on the
   * same row)
   */
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
  rafko_net::SolutionSolver::Factory reference_solver_factory(network,
                                                              settings);
  std::cout << "Optimizing network:" << std::endl;
  std::cout << "Training Error; \t\tTesting Error; min; \t\t avg_d_w_abs; \t\t "
               "iteration; \t\t duration(ms); avg duration(ms)\t "
            << std::endl;
  while (
      !optimizer->stop_triggered()) { /* If Early stopping or zero-error
                                         condition is set, the training quits */
    std::shared_ptr<rafko_net::SolutionSolver> reference_solver =
        reference_solver_factory.build();

    /* One Trianing iteration is called, and the runtime is evaluated */
    start = std::chrono::steady_clock::now();
    optimizer->iterate();
    auto current_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (0.0 == avg_duration)
      avg_duration = current_duration;
    else
      avg_duration = (avg_duration + current_duration) / 2.0;

    /* Reading out test and training set data */
    train_error = optimizer->get_last_training_error();
    test_error = optimizer->get_last_testing_error();
    if (abs(test_error) < minimum_error) {
      minimum_error = abs(test_error);
      std::cout << std::endl;
    }

    /* displaying the results */
    std::cout << "\r";
    for (std::uint32_t space_count = 0; space_count < console_width - 1;
         ++space_count)
      std::cout << " ";
    std::cout << "\r";
    std::cout << std::setprecision(9) << train_error << ";\t\t" << test_error
              << "; " << minimum_error << ";\t\t"
              << optimizer->get_avg_of_abs_gradient() << ";\t\t" << iteration
              << ";\t\t" << current_duration << "; " << avg_duration << "; "
              << std::flush;
    ++iteration;
    if (std::abs(test_error) <= low_error) {
      iteration_reached_low_error =
          std::min(iteration_reached_low_error, iteration);
      if ((iteration - iteration_reached_low_error) > iteration_optimum_delta) {
        std::cout << std::endl << "== good enough for a test ==" << std::endl;
        break; /* End the loop if enough loops spent below error threshold */
      }
    }
  }
  std::cout << std::endl
            << "Optimum reached in " << (iteration + 1)
            << " iterations!(average runtime: " << avg_duration << " ms)   "
            << std::endl;

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid
                                                  false positive memory leak
                                                  reports in memory leak
                                                  analyzers */
  return 0;
}
