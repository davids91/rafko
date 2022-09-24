#include <iostream>
#include <memory>

#include <google/protobuf/arena.h>
#include <rafko_protocol/rafko_net.pb.h>
#include <rafko_mainframe/models/rafko_settings.hpp>
#include <rafko_net/services/rafko_net_builder.hpp>

int main(){
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    /* +++ Standard Training settings +++ */
    .set_learning_rate(2e-4)
    .set_learning_rate_decay({{100u,0.8},{500u,0.5}}) /* multiplier applied at the given training iterations */
    .set_minibatch_size(64)
    .set_memory_truncation(2) /* limit gradient calculation in case of sequential data */
    .set_droput_probability(0.0)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero, true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping, false)
    .set_tolerance_loop_value(10)
    .set_delta(0.1) /* for early stopping */

    /* +++ Environmental/Computational settings +++ */
    /*!Note: .set_arena_ptr(&arena) is optional, because @rafko_gym::RafkoAutodiffOptimizer
     * sets its own internally if an arena is not already present inside the rafko_net::RafkoSettings instance.
     */
    .set_max_solve_threads(2).set_max_processing_threads(4)
    /*!Note: max_solve_threads suggests the number of solutions to be processed in paralell,
     * while max_processing_threads suggests maximum number of threads processing each.
     * The ideal number for this: number of CPU cores = max_processing_threads * max_solve_threads
     * These settings are for CPU only, as GPU training / processing works with a significantly different core(worker) count.
     */
  );

  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(2).expected_input_range(1.0)
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_boltzmann_knot)
    .add_feature_to_layer(1u, rafko_net::neuron_group_feature_boltzmann_knot)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .dense_layers({3,2,1});

  //TODO: Training in opencl and not in opencl

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid false positive memory leak reports in memory leak analyzers */
  return 0;
}
