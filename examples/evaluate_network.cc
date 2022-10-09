#include <iostream>
#include <memory>
#include <cmath>

#include <google/protobuf/arena.h>
#include <rafko_protocol/rafko_net.pb.h>
#include <rafko.hpp>

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
  std::unique_ptr<rafko_mainframe::RafkoGPUContext> second_context = (
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(*network, settings, objective)
  );
  #else
  std::unique_ptr<rafko_mainframe::RafkoCPUContext> second_context = std::make_unique<rafko_mainframe::RafkoCPUContext>(
    *network, settings, objective
  );
  #endif/*(RAFKO_USES_OPENCL)*/

  /*!Note: OpenCL context provides speed, which is only visible when evaluating or running many data at once.
   * For smaller batch sizes copying the data to the GPU and back comes with a big overhead compared to the speedup
   * from paralellization. To make use of OpenCL, an environment can be solved at once like below,
   * but the provided vector needs to have the exact sizes of the relevant(more on this later) output of the environment.
   */
  second_context->set_environment(environment);
  std::vector<std::vector<double>> environment_result(
    environment->get_number_of_label_samples(), std::vector<double>(network->output_neuron_number())
  );
  second_context->solve_environment(environment_result);
  double first_batch_result = environment_result[0][0];

  /*!Note: the above however does not have any data from previous runs */
  second_context->solve_environment(environment_result);
  assert(first_batch_result == environment_result[0][0]);

  /*!Note: to have the context remember the results of the previous runs ( up to the memory of the network)
   * non-isolated batch-solve can be used as below.
   */
  #if(!RAFKO_USES_OPENCL) /* In case of CPU context, the buffers available are that of the number of threads offered by the CPU */
  environment_result.resize( std::min(
    environment->get_number_of_label_samples(), settings->get_max_processing_threads()
  ) );
  #endif/*(RAFKO_USES_OPENCL)*/
  second_context->solve_environment(environment_result, true/*isolated*/);
  assert(first_batch_result == environment_result[0][0]);
  second_context->solve_environment(environment_result, false/*isolated*/);
  assert(first_batch_result != environment_result[0][0]);

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid false positive memory leak reports in memory leak analyzers */
  return 0;
}
