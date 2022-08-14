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

#include "rafko_global.hpp"

#include <memory>
#include <mutex>
#include <vector>
#include <CL/opencl.hpp>

#include "rafko_net/services/solution_solver.hpp"
#include "rafko_gym/models/rafko_environment.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/services/rafko_weight_adapter.hpp"
#include "rafko_gym/services/updater_factory.hpp"

#include "rafko_mainframe/services/rafko_gpu_phase.hpp"
#include "rafko_mainframe/services/rafko_context.hpp"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoGPUContext : public RafkoContext{
public:
  RafkoGPUContext(
    cl::Context&& context_, cl::Device device_,
    rafko_mainframe::RafkoSettings settings_, rafko_net::RafkoNet& neural_network_,
    std::shared_ptr<rafko_gym::RafkoObjective> objective_
  );

  /* +++ Methods taken from @RafkoContext +++ */
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_) override;
  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_) override;
  void set_weight_updater(rafko_gym::Weight_updaters updater) override;

  void refresh_solution_weights() override{
    RFASSERT_LOG("Refreshing Solution weights in CPU context..");
    weight_adapter.update_solution_with_weights();
    upload_weight_table_to_device();
  }

  void set_network_weight(std::uint32_t weight_index, double weight_value) override;
  void set_network_weights(const std::vector<double>& weights) override;
  void apply_weight_update(const std::vector<double>& weight_delta) override;
  double full_evaluation() override;
  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u) override;

  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input,
    bool reset_neuron_data = false, std::uint32_t thread_index = 0
  ) override;

  void push_state() override{
    environment->push_state();
  }

  void pop_state() override{
    environment->pop_state();
  }

  rafko_mainframe::RafkoSettings& expose_settings() override{
    last_ran_evaluation = not_eval_run; /* in case some training parameters changed buffers might need to be refreshed */
    return settings;
  }

  rafko_net::RafkoNet& expose_network() override{
    return network;
  }
  /* --- Methods taken from @RafkoContext --- */

  ~RafkoGPUContext() = default;

private:
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
  bool last_random_eval_was_seeded = false;
  std::uint32_t last_used_seed;
  /* Somebody tell me what is the least propable value of a random seed one can use,
   * so I could initialize this poor fella with it?!
   */

  enum{
    not_eval_run, full_eval_run, random_eval_run
  }last_ran_evaluation = not_eval_run;

  /**
   * @brief   Uploads the weights from @network to the buffer on the GPU
   */
  void upload_weight_table_to_device();

  /**
   * @brief   Uploads the weights from @network to the buffer on the GPU for the given weight index
   *
   * @param[in]     weight_index    the index of the weight to upload
   */
  void upload_weight_to_device(std::uint32_t weight_index);

  /**
   * @brief     sets the paramterers of the objective based on the environment, re-generates its kernels and uploads them to GPU
   */
  void refresh_objective();

  /**
   * @brief     Upload inputs to the solution phase to be able to run the agent kernel code on the inputs
   *
   * @param[in]   sequence_start_index          The index of the first sequence in the environment to upload the inputs from
   * @param[in]   buffer_sequence_start_index   Start index of a sequence to start uploading inputs from in the global buffer
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   *
   * @return      A vector of events to wait for, signaling operation completion
   */
  [[nodiscard]] std::vector<cl::Event> upload_agent_output(
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
  [[nodiscard]] double error_post_process(double raw_error, std::uint32_t labels_evaluated);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_GPU_CONTEXT_H */
