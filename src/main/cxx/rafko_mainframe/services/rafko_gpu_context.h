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

#ifndef RAFKO_CPU_CONTEXT_H
#define RAFKO_CPU_CONTEXT_H

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <CL/opencl.hpp>

#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_agent.h"
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_gpu_phase.h"
#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoGPUContext : public RafkoContext{
public:
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_);

  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_){
    objective = objective_;
  }

  void set_weight_updater(rafko_gym::Weight_updaters updater){
    weight_updater = rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, updater, settings);
  }

  void set_network_weight(uint32 weight_index, sdouble32 weight_value);
  void set_network_weights(const std::vector<sdouble32>& weights);
  void apply_weight_update(const std::vector<sdouble32>& weight_delta);
  sdouble32 full_evaluation();
  sdouble32 stochastic_evaluation(bool to_seed = false, uint32 seed_value = 0u);
  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<sdouble32>& input,
    bool reset_neuron_data = true, uint32 thread_index = 0
  );

  void push_state(){
    environment->push_state();
  }

  void pop_state(){
    environment->pop_state();
  }

  rafko_mainframe::RafkoSettings& expose_settings(){
    return settings;
  }

  rafko_net::RafkoNet& expose_network(){
    return network;
  }
  ~RafkoGPUContext() = default;

  class Builder{
  public:
    Builder(rafko_net::RafkoNet neural_network_, rafko_mainframe::RafkoSettings settings_ = rafko_mainframe::RafkoSettings());
    Builder& select_platform(uint32 platform_index = 0u);
    Builder& select_device(cl_device_type type = CL_DEVICE_TYPE_GPU, uint32 device_index = 0u);
    std::unique_ptr<RafkoGPUContext> build();

  private:
    rafko_mainframe::RafkoSettings settings;
    rafko_net::RafkoNet network;
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    uint32 selected_platform = 0u;
    uint32 selected_device = 0u;
  };
private:

  RafkoGPUContext(
    cl::Context&& context_, cl::Device&& device_,
    rafko_mainframe::RafkoSettings&& settings_, rafko_net::RafkoNet&& neural_network_
  );

  google::protobuf::Arena arena;
  rafko_mainframe::RafkoSettings settings;
  rafko_net::RafkoNet network;
  std::unique_ptr<rafko_net::Solution> network_solution;
  std::unique_ptr<rafko_net::SolutionSolver> agent;
  std::shared_ptr<rafko_gym::RafkoEnvironment> environment;
  std::shared_ptr<rafko_gym::RafkoObjective> objective;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;
  std::vector<std::vector<sdouble32>> neuron_outputs_to_evaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup execution_threads;

  cl::Context opencl_context;
  cl::Device opencl_device;
  cl::CommandQueue opencl_queue;
  cl::Buffer weights_and_inputs;
  cl::Buffer features_and_labels;
  cl::Buffer error_value;
  RafkoGPUPhase solution_phase;
  RafkoGPUPhase error_phase;

  uint32 used_sequence_truncation;
  uint32 used_minibatch_size;

};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CPU_CONTEXT_H */
