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

#ifndef RAFKO_GPU_CONTEXT_H
#define RAFKO_GPU_CONTEXT_H

#include "rafko_global.h"

#include <memory>
#include <mutex>
#include <vector>
#include <CL/opencl.hpp>

#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_agent.h"
#include "rafko_gym/services/rafko_weight_adapter.h"
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_gpu_phase.h"
#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoGPUContext : public RafkoContext{
public:

  void fix_dirty(){ /*!Note: When weights are updated elsewhere this hack takes over the changes */
    weight_adapter.update_solution_with_weights();
    upload_weight_table_to_device();
  }

  /* +++ Methods taken from @RafkoContext +++ */
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_);
  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_);
  void set_weight_updater(rafko_gym::Weight_updaters updater);

  void refresh_solution_weights(){
    RFASSERT_LOG("Refreshing Solution weights in CPU context..");
    weight_adapter.update_solution_with_weights();
    upload_weight_table_to_device();
  }

  void set_network_weight(std::uint32_t weight_index, double weight_value);
  void set_network_weights(const std::vector<double>& weights);
  void apply_weight_update(const std::vector<double>& weight_delta);
  double full_evaluation();
  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u);

  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input,
    bool reset_neuron_data = false, std::uint32_t thread_index = 0
  );

  void push_state(){
    environment->push_state();
  }

  void pop_state(){
    environment->pop_state();
  }

  constexpr rafko_mainframe::RafkoSettings& expose_settings(){
    last_ran_evaluation = not_eval_run; /* in case some training parameters changed buffers might need to be refreshed */
    return settings;
  }

  constexpr rafko_net::RafkoNet& expose_network(){
    return network;
  }
  /* --- Methods taken from @RafkoContext --- */

  ~RafkoGPUContext() = default;

  class Builder{
  public:
    Builder(rafko_net::RafkoNet& neural_network_, rafko_mainframe::RafkoSettings settings_ = rafko_mainframe::RafkoSettings());
    Builder& select_platform(std::uint32_t platform_index = 0u);
    Builder& select_device(cl_device_type type = CL_DEVICE_TYPE_GPU, std::uint32_t device_index = 0u);
    std::unique_ptr<RafkoGPUContext> build();
  private:
    rafko_mainframe::RafkoSettings settings;
    rafko_net::RafkoNet& network;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::uint32_t selected_platform = 0u;
    std::uint32_t selected_device = 0u;
    RFASSERT_SCOPE(RAFKO_GPU_BUILD);
  };
private:

  RafkoGPUContext(
    cl::Context&& context_, cl::Device&& device_,
    rafko_mainframe::RafkoSettings&& settings_, rafko_net::RafkoNet& neural_network_
  );

  rafko_net::RafkoNet& network;
  std::unique_ptr<rafko_net::Solution> network_solution;
  rafko_gym::RafkoWeightAdapter weight_adapter;
  std::shared_ptr<rafko_net::SolutionSolver> agent;
  std::shared_ptr<rafko_gym::RafkoEnvironment> environment;
  std::shared_ptr<rafko_gym::RafkoObjective> objective;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;
  std::vector<std::vector<double>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup execution_threads;

  cl::Context opencl_context;
  cl::Device opencl_device;
  cl::CommandQueue opencl_queue;
  std::uint32_t device_weight_table_size;
  RafkoGPUPhase solution_phase;
  std::vector<double> standalone_solution_result;
  RafkoGPUPhase error_phase;

  enum{
    not_eval_run, full_eval_run, random_eval_run
  }last_ran_evaluation = not_eval_run;

  void upload_weight_table_to_device();
  void upload_weight_to_device(std::uint32_t weight_index);
  void refresh_objective();
  bool last_random_eval_was_seeded = false;
  std::uint32_t last_used_seed;
  /* Somebody tell me what is the least propable value of a random seed one can use,
   * so I could initialize this poor fella with it?!
   */

  /**
   * @brief     Upload inputs to the solution phase to be able to run the agent kernel code on the inputs
   *
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  std::vector<cl::Event> upload_agent_inputs(
    std::uint32_t sequence_start_index, std::uint32_t buffer_sequence_start_index, std::uint32_t sequences_to_upload
  );

  /**
   * @brief     Upload labels to the error phase to be able to evaluate agent output
   *
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   * @param[in]   buffer_start_byte_offset      The start index where the labels start inside the evaluation buffer
   * @param[in]   start_index_inside_sequence   Start index inside sequence for sequence truncation
   * @param[in]   sequence_truncation           Number of labels to evaluate per sequence (sequence truncation size)
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  std::vector<cl::Event> upload_labels(
    std::uint32_t sequence_start_index, std::uint32_t buffer_sequence_start_index,
    std::uint32_t sequences_to_upload, std::uint32_t buffer_start_byte_offset,
    std::uint32_t start_index_inside_sequence, std::uint32_t sequence_truncation
  );

  /**
   * @brief     Upload inputs to the solution phase to be able to run the agent kernel code on the inputs
   *
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  std::vector<cl::Event> upload_agent_output(
    std::uint32_t sequences_to_upload, std::uint32_t start_index_inside_sequence, std::uint32_t sequence_truncation
  );

  /**
   * @brief     Process raw error value by adding the performance feature related errors
   *            and dividing the result by the number of evaluated labels
   *
   * @param[in]   raw_error           The raw error from the evaluation
   * @param[in]   labels_evaluated    The number fo labels which resulted in the raw error
   *
   * @return    The processed error
   */
  double error_post_process(double raw_error, std::uint32_t labels_evaluated);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_CONTEXT_H */
