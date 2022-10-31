#include <iostream>
#include <memory>
#include <cmath>
#include <thread>
#include <iostream>
#include <fstream>

#include <rafko_protocol/rafko_net.pb.h>
#include <rafko.hpp>

int main(){
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>();
  settings->set_arena_ptr(&arena);

  /* Create a Densely Connected Neural Network */
  using rafko_net::RafkoNet;
  RafkoNet* network = (
    rafko_net::RafkoNetBuilder(*settings)
    .input_size(2)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_relu, rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .create_layers({3, 3, 1})
  );

  /*!Note: To solve any network with Rafko on CPU Rafko implements the below class and Factories */
  rafko_net::SolutionSolver::Factory solverFactory(*network, settings);
  std::shared_ptr<rafko_net::SolutionSolver> solver_from_factory = solverFactory.build();

  /*!Note: If an arena pointer is set in the settings: The intermediate representation of the network,
   * which @rafko_net::SolutionSolver uses is owned by it. In that case the factory might go out of scope,
   * and the solver would remain functional.
   */
  std::shared_ptr<rafko_net::SolutionSolver> tinkered_network_solver = rafko_net::SolutionSolver::Factory(*network, settings).build();

  /*!Note: Solving the network gives const access to its internal buffer through the returned result.
   * It's a rough equivalent of a C++20 std::span; So it can be iterated and read as a std::vector would be.
   * The size of it is the output of the network, which is the size of the éast layer,
   * or set by @output_neuron_number in @rafko_net::RafkoNetBuilder
   */
  rafko_utilities::ConstVectorSubrange<> original_result_reference = solver_from_factory->solve({1.0, 2.0});

  /*!Note: Since the returned object is a const reference, its value might change as the network is solved multiple times.
   * a copy of the value can be acquired like below.
   */
  std::vector<double> original_result = original_result_reference.acquire();
  for([[maybe_unused]]const double& output : original_result){ /*...*/ }
  assert(!std::isnan(original_result[0]));

  /*!Note: since Neuron data might be stored in a buffer as long as the network memory buffer permits it,
   * So it might be accessed later by the network. This essentially means multiple run produces different results
   * whenever neuron recurrence is used (@add_neuron_recurrence) or a layer has the feature rafko_net::neuron_group_feature_boltzmann_knot
   * activated. To Reset the internal Neuron buffers, an optional argument may be passed to @solve. By default the bufers are not
   * resetted before every run.
   */
  auto second_result = solver_from_factory->solve({1.0, 2.0}/*input*/, false/*reset_neuron_data*/).acquire();
  assert(second_result[0] != original_result[0]);
  assert(original_result[0] == solver_from_factory->solve({1.0, 2.0}, true)[0]); /*!Note: This call resets Neuron buffers */

  /*!Note: rafko_net::SolutionSolver is thread-safe, usable up to the number of threads
   * set by @get_max_processing_threads in @rafko_mainframe::RafkoSettings.
   * Each thread has their own buffer for Neuron values.
   */
  double paralell_result;
  std::thread paralell_solve_thread = std::thread([&paralell_result, &solver_from_factory](){
    paralell_result = solver_from_factory->solve({1.0, 2.0}, false, 1u/*thread_index*/)[0];
  });
  double another_result = solver_from_factory->solve({1.0, 2.0}, 0u/*thread_index*/)[0];
  paralell_solve_thread.join();

  assert(original_result[0] == paralell_result);
  assert(second_result[0] == another_result);

  /*!Note: Other Solvers can be built with the Factory interface for the same network. */
  std::shared_ptr<rafko_net::SolutionSolver> another_solver_from_the_same_factory = solverFactory.build();

  /*!Note: Changing the weights of the network doesn't affect the solvers directly,
   * as they don't handle the intermediate representation(@rafko_net::Solution) of it.
   * An intermediate representation is owned by the arena(if the pointer is set in settings) if present
   * and handled by the Factory: It is destroyed if an arena is not set in settings, and weight update
   * can be triggered by it.
   */
  for(double& w : *network->mutable_weight_table()){ w = 0.5; }
  solverFactory.refresh_actual_solution_weights(); /* Update the weight values of the Solution */
  auto result_with_changed_weights = solver_from_factory->solve({1.0, 2.0}, true).acquire();
  auto another_result_with_changed_weights = another_solver_from_the_same_factory->solve({1.0, 2.0}, true).acquire();
  assert(result_with_changed_weights[0] == another_result_with_changed_weights[0]);
  assert(another_result_with_changed_weights[0] != original_result[0]);

  /*!Note: Changing the network structure directly, even in non-quantitative ways makes the solvers obsolete,
   * because they store intermediate representation(@Solution) references internally. New Solvers can be constructed
   * through the factory interface. Setting the @refresh_solution parameter to true makes the factory follow
   * structural changes as well; Depending on network size it might consume a significant amount of runtime.
   */
  network->mutable_neuron_array(0)->set_transfer_function(rafko_net::transfer_function_sigmoid);

  /*!Note: Solvers, however remain functional, as duplicates are produced upon each update of the
   * Intermediate solution. BEWARE: RAM is limited, so building new Solutions frequently is discouraged, as
   * each new build takes up additional space in the Arena.
   */
  std::shared_ptr<rafko_net::SolutionSolver> updated_solver = solverFactory.build(true/*rebuild_solution*/);

  auto changed_solution_result = updated_solver->solve({1.0, 2.0}).acquire();
  auto unchanged_solution_result = solver_from_factory->solve({1.0, 2.0}, true).acquire();
  assert(changed_solution_result[0] != unchanged_solution_result[0]);

  /*!Note: To handle Solutions in a more storage-friendly manner the factory can be used to replace the last created Solution.
   * It however only replaces the lat built Solution, so Solvers built based on previous Solutions are not updated.
   */
  network->mutable_neuron_array(0)->set_transfer_function(rafko_net::transfer_function_identity);
  std::shared_ptr<rafko_net::SolutionSolver> another_updated_solver = solverFactory.build(
    true/*rebuild_solution*/, true/*swap_solution*/
  );
  auto latest_result = another_updated_solver->solve({1.0, 2.0}).acquire();
  auto also_latest_result = updated_solver->solve({1.0, 2.0}, true).acquire(); /* the buffers needs to be cleared to have an identical result */
  assert(changed_solution_result[0] != latest_result[0]);
  assert(also_latest_result[0] == latest_result[0]);

  /*!Note: Older solvers keep their own Solution objects, and are not managed by the factory,
   * once the factory builds a new version of a Solution with (rebuild_solution == true && swap_solution == false)
   */
  assert(result_with_changed_weights[0] == solver_from_factory->solve({1.0, 2.0}, true).acquire()[0]);

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid false positive memory leak reports in memory leak analyzers */
  return 0;
}
