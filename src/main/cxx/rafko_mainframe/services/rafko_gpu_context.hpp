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
#include <functional>
#include <CL/opencl.hpp>

#include "rafko_net/services/solution_solver.hpp"
#include "rafko_gym/models/rafko_environment.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/services/updater_factory.hpp"

#include "rafko_mainframe/services/rafko_gpu_phase.hpp"
#include "rafko_mainframe/services/rafko_context.hpp"

namespace rafko_mainframe {

class RAFKO_EXPORT RafkoGPUContext : public RafkoContext{
public:
  RafkoGPUContext(
    cl::Context&& context, cl::Device device,
    rafko_net::RafkoNet& neural_network,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {},
    std::shared_ptr<rafko_gym::RafkoObjective> objective = {}
  );

  /* +++ Methods taken from @RafkoContext +++ */
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment) override;
  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective) override;
  void set_weight_updater(rafko_gym::Weight_updaters updater) override;
  void set_network_weight(std::uint32_t weight_index, double weight_value) override;
  void set_network_weights(const std::vector<double>& weights) override;
  void apply_weight_update(const std::vector<double>& weight_delta) override;
  double full_evaluation() override;
  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u) override;

  void refresh_solution_weights() override{
    RFASSERT_LOG("Refreshing Solution weights in CPU context..");
    m_solverFactory.refresh_actual_solution_weights();
    upload_weight_table_to_device();
  }

  void push_state() override{
    m_environment->push_state();
  }

  void pop_state() override{
    m_environment->pop_state();
  }

  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input,
    bool reset_neuron_data = false, std::uint32_t thread_index = 0
  ) override;

  void solve_environment(std::vector<std::vector<double>>& output) override;

  rafko_mainframe::RafkoSettings& expose_settings() override{
    m_lastRanEvaluation = not_eval_run; /* in case some training parameters changed buffers might need to be refreshed */
    return *m_settings;
  }

  rafko_net::RafkoNet& expose_network() override{
    return m_network;
  }
  /* --- Methods taken from @RafkoContext --- */

  ~RafkoGPUContext() = default;

private:
  rafko_net::RafkoNet& m_network;
  rafko_net::SolutionSolver::Factory m_solverFactory;
  std::shared_ptr<rafko_net::SolutionSolver> m_agent;
  std::shared_ptr<rafko_gym::RafkoEnvironment> m_environment;
  std::shared_ptr<rafko_gym::RafkoObjective> m_objective;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> m_weightUpdater;
  std::vector<std::vector<double>> m_neuronOutputsToEvaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup m_executionThreads;

  cl::Context m_openclContext;
  cl::Device m_openclDevice;
  cl::CommandQueue m_openclQueue;
  std::uint32_t m_deviceWeightTableSize;
  RafkoGPUPhase m_solutionPhase;
  std::vector<double> m_standaloneSolutionResult;
  RafkoGPUPhase m_errorPhase;
  bool m_lastRandomEvalWasSeeded = false;
  std::uint32_t m_lastUsedSeed;
  /* Somebody tell me what is the least propable value of a random seed one can use,
   * so I could initialize this poor fella with it?!
   */
  std::uint32_t m_numOutputsInOneSequence = 1u;
  std::uint32_t m_evalStartInSequence = 0u;

 enum{
   not_eval_run, full_eval_run, random_eval_run
 }m_lastRanEvaluation = not_eval_run;

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
   * @param[in]   sequences_to_upload           The number of sequences to upload the inputs from
   * @param[in]   start_index_inside_sequence   Start index of a feature vector inside every sequence to start uploading inputs from
   *                                            Index 0 starts from the first feature vector assigned for each sequence
   *                                            In case the Network has more memory, than the environment label pairs: index 0 starts at zero still.
   *                                            In that case, the evaluation doesn't start form 0, as label and prefill values take up less space than what is assigned for one sequence
   * @param[in]   sequence_truncation           Number of feature vectors to upload per sequence
   * @param[in]   data_handler                  The funciton accepting the CL Buffer, byte offset, and data size(bytes == output neurons only!) for each feature for one sequence
   */
  void upload_agent_output(
    std::uint32_t sequences_to_upload, std::uint32_t start_index_inside_sequence, std::uint32_t sequence_truncation,
    std::function<void(cl::Buffer, std::uint32_t, std::uint32_t)> data_handler
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
