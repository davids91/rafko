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

#ifndef RAFKO_AUTODIFF_GPU_OPTIMIZER_H
#define RAFKO_AUTODIFF_GPU_OPTIMIZER_H

#include "rafko_global.h"

#include <cmath>
#include <memory>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#include "rafko_mainframe/services/rafko_gpu_phase.h"
#include "rafko_mainframe/services/rafko_dummies.h"
#include "rafko_gym/services/rafko_autodiff_gpu_strategy.h"
#include "rafko_gym/services/rafko_autodiff_optimizer.h"

namespace rafko_gym{

/**
 * @brief
 */
class RAFKO_FULL_EXPORT RafkoAutodiffGPUOptimizer
: private RafkoAutodiffOptimizer
{
public:
  RafkoAutodiffGPUOptimizer(
    cl::Context&& context_, cl::Device device_,
    const rafko_mainframe::RafkoSettings& settings_,
    std::shared_ptr<RafkoEnvironment> environment_, rafko_net::RafkoNet& network_,
    std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator_ = {},
    std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator_ = {}
  )
  : RafkoAutodiffOptimizer(settings_, environment_, network_, training_evaluator_, test_evaluator_)
  , opencl_context(context_)
  , opencl_device(device_)
  , opencl_queue(opencl_context, opencl_device)
  , strategy(std::make_shared<AutoDiffGPUStrategy>(settings, network, environment))
  , gpu_phase(
    opencl_context, opencl_device, opencl_queue,
    std::make_shared<rafko_mainframe::RafkoDummyGPUStrategyPhase>(
      rafko_mainframe::RafkoNBufShape({0u})/*input_shape*/,
      rafko_mainframe::RafkoNBufShape({0u})/*output_shape*/
    )
  )
  {
  }

  void build(std::shared_ptr<RafkoObjective> objective);
  using RafkoAutodiffOptimizer::set_weight_updater;
  using RafkoAutodiffOptimizer::stop_triggered;
  using RafkoAutodiffOptimizer::get_last_training_error;
  using RafkoAutodiffOptimizer::get_last_testing_error;
  using RafkoAutodiffOptimizer::get_avg_gradient;
  void iterate(bool refresh_environment = false);

  /**
   * @brief     Uploads the weight table from the network into its internal buffers
   */
  void upload_weight_table();

  /**
   * @brief     Uploads the input data from the environment into its internal buffers
   *
   * @return    A vector of events signaling when the operations are ready
   */
  std::vector<cl::Event> update_inputs();

  /**
   * @brief     Uploads the label data from the environment into its internal buffers
   *
   * @return    A vector of events signaling when the operations are ready
   */
  std::vector<cl::Event> update_labels();

  /**
   * @brief     Refreshes buffer data based on its current status
   */
  void refresh_GPU_environment();


  double get_neuron_data(
    std::uint32_t sequence_index, std::uint32_t past_index, std::uint32_t neuron_index
  );

private:
  cl::Context opencl_context;
  cl::Device opencl_device;
  cl::CommandQueue opencl_queue;
  std::shared_ptr<AutoDiffGPUStrategy> strategy;
  rafko_mainframe::RafkoGPUPhase gpu_phase;

};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OCL_OPTIMIZER_H */
