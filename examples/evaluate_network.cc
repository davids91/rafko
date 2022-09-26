#include <iostream>
#include <memory>
#include <cmath>

#include <google/protobuf/arena.h>
#include <rafko_protocol/rafko_net.pb.h>
#include <rafko_net/services/rafko_net_builder.hpp>
#include <rafko_gym/models/rafko_cost.hpp>
#include <rafko_gym/models/rafko_dataset_wrapper.hpp>
#include <rafko_mainframe/models/rafko_settings.hpp>
#include <rafko_mainframe/services/rafko_cpu_context.hpp>

#if(RAFKO_USES_OPENCL)
#include <rafko_mainframe/services/rafko_ocl_factory.hpp>
#include <rafko_mainframe/services/rafko_gpu_context.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

int main(){
  google::protobuf::Arena arena;
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    /* +++ Environmental/Computational settings +++ */
    .set_arena_ptr(&arena)
    /*!Note: .set_arena_ptr(&arena) might be optional, because @rafko_mainframe::RafkoCPUContext
     * sets its own internally if an arena is not already present inside the rafko_net::RafkoSettings instance.
     */
    .set_max_solve_threads(2) /* How many network solve processes to run in paralell */
    .set_max_processing_threads(4) /* How many processes to run for each solve process */
    /*!Note: max_solve_threads suggests the number of solutions to be processed in paralell,
     * while max_processing_threads suggests maximum number of threads processing each.
     * The ideal number for this: number of CPU cores = max_processing_threads * max_solve_threads
     * These settings are for CPU only, as GPU training / processing works with a significantly different core(worker) count.
     */

     /* +++ Evaluation settings +++ */
    .set_minibatch_size(64) /* How many sequences of feature-label pairs to evaluate in one stochastic evaluation */
    .set_memory_truncation(2) /* How many feautre-label pairs to evaluate inside a sequence of them */
  );

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
    .dense_layers({3, 3, 1})
  );

  /*!Note: @RafkoContext can be used to handle a network Intermediate representation and solver.
   * It requires only a network. Settings and an Objective ( for evaluation ) are optional.
   */
  rafko_mainframe::RafkoCPUContext context_with_own_settings(*network);
  rafko_mainframe::RafkoCPUContext context(*network, settings);
  auto context_result = context.solve({1.0, 2.0}).acquire();

  /*!Note: Contexts handle evaluation as well, if the provided parameters are set:
  * - Environment: A set of feature-label pairs to evaluate the network on
  *    \--> in deep learning jargon this equates to a data set
  * - Objective: The Object calculating the error value from the difference between the feature and label sets
  *    \--> in Deep learning jargon
  */
  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_mse
  );
  context.set_objective(objective);

  /* +++ Generating sample environment Data set +++ */
  constexpr const std::uint32_t sequence_size = 5u;
  std::vector<std::vector<double>> environment_inputs;
  std::vector<std::vector<double>> environment_labels;
  for(std::uint32_t sequence_index = 0; sequence_index < settings->get_minibatch_size() * 3; ++sequence_index){
    /*!Note: For this example, since the values don't matter full zero content is fine */
    for(std::uint32_t datapoint_index = 0; datapoint_index < sequence_size; ++datapoint_index){
      environment_inputs.push_back(std::vector<double>(network->input_data_size()));
      environment_labels.push_back(std::vector<double>(network->output_neuron_number()));
      /*!Note: It is possible to add more inputs, than the networks input size here and still have a valid environment
       * Inside each sequence, the number of labels are compared to the size of the sequence, and surplus inputs
       * at the beginning of each sequence are counted as "prefill". Their only purpose is to initialize
       * the networks data buffers for each sequence, np error scores or gradients are calculated from them
       */
    }
  }
  /* --- Generating sample environment Data set --- */
  std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(
    std::move(environment_inputs), std::move(environment_labels), sequence_size
  );
  context.set_environment(environment);

  /*!Note: Evaluation results may be given in both positive and negative ranges.
   * Negative number represents an error value, while positive value represents a fitness value.
   * Fittness values may be used with numeric optimizers.
   */
  double full_evaluation_result = context.full_evaluation();
  double stochastic_evaluation_result = context.stochastic_evaluation();
  assert(stochastic_evaluation_result > full_evaluation_result);

  /*!Note: to build a context using OpenCL the following factory can be used */
  #if(RAFKO_USES_OPENCL)
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> opencl_context = (
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(*network, settings, objective)
  );
  #endif/*(RAFKO_USES_OPENCL)*/

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid false positive memory leak reports in memory leak analyzers */
  return 0;
}
