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

#include "rafko_global.hpp"

#include <cmath>
#include <memory>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"
#include "rafko_mainframe/services/rafko_gpu_phase.hpp"
#include "rafko_mainframe/services/rafko_dummies.hpp"
#include "rafko_gym/services/rafko_autodiff_gpu_strategy.hpp"
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

namespace rafko_gym{

/**
 * @brief   A class to calculate the values and derivatives of a Network, and update its weights based on it
 *          using OpenCL drivers
 */
class RAFKO_EXPORT RafkoAutodiffGPUOptimizer : public RafkoAutodiffOptimizer
{
public:
  RafkoAutodiffGPUOptimizer(
    cl::Context&& context, cl::Device device,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings,
    rafko_net::RafkoNet& network, std::shared_ptr<RafkoDataSet> data_set = {}, 
    std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator = {},
    std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator = {}
  )
  : RafkoAutodiffOptimizer(settings, network, training_evaluator, test_evaluator)
  , m_openclContext(context)
  , m_openclDevice(device)
  , m_openclQueue(m_openclContext, m_openclDevice)
  , m_strategy(std::make_shared<AutoDiffGPUStrategy>(*m_settings, m_network, data_set))
  , m_gpuPhase(
    m_openclContext, m_openclDevice, m_openclQueue,
    std::make_shared<rafko_mainframe::RafkoDummyGPUStrategyPhase>(
      rafko_mainframe::RafkoNBufShape({0u})/*input_shape*/,
      rafko_mainframe::RafkoNBufShape({0u})/*output_shape*/
    )
  )
  {
  }

  void build(const std::shared_ptr<RafkoDataSet> data_set, std::shared_ptr<RafkoObjective> objective) override;
  using RafkoAutodiffOptimizer::set_weight_updater;
  using RafkoAutodiffOptimizer::stop_triggered;
  using RafkoAutodiffOptimizer::get_last_training_error;
  using RafkoAutodiffOptimizer::get_last_testing_error;
  using RafkoAutodiffOptimizer::get_avg_of_abs_gradient;
  using RafkoAutodiffOptimizer::apply_weight_update;
  using RafkoAutodiffOptimizer::update_context_errors;
  using RafkoAutodiffOptimizer::set_training_context;
  using RafkoAutodiffOptimizer::set_testing_context;

  /**
   * @brief   calculate the values and derivatives and update the weights based on them. The sizes of the buffers are supposed based on the provided 
   *          @data_set reference, but actual data uploaded to the GPU from the @data_set only if @refresh_data_set is true
   *
   * @param[in]   data_set            The data set the network is evaluated on
   * @param[in]   force_gpu_upload    if true, the  input values and labels will be reuploaded to GPU based on the dependencies
   */
  void iterate(const RafkoDataSet& data_set, bool force_gpu_upload = false) override;

  /**
   * @brief   provides the average gradient for the weight under the given index
   * 
   * @return    the average weight gradient for the weight under the given weight index
   */
  double get_avg_gradient(std::uint32_t d_w_index) const override;

  /**
   * @brief     Uploads the weight table from the network into its internal buffers
   */
  void upload_weight_table();

  /**
   * @brief     Uploads the input data from the data set into its internal buffers
   *
   * @param[in]   data_set                The data set the network is evaluated on
   * 
   * @return    A vector of events signaling when the operations are ready
   */
  [[nodiscard]] std::vector<cl::Event> update_inputs(const RafkoDataSet& data_set);

  /**
   * @brief     Uploads the label data from the data set into its internal buffers
   *
   * @param[in]   data_set                The data set the network is evaluated on
   * 
   * @return    A vector of events signaling when the operations are ready
   */
  [[nodiscard]] std::vector<cl::Event> update_labels(const RafkoDataSet& data_set);

  /**
   * @brief     Refreshes buffer data based on its current status
   * 
   * @param[in]   data_set    The data set the network is evaluated on
   */
  void sync_data_set_on_GPU(const RafkoDataSet& data_set);

  /**
   * @brief     Downloads the activation value of a single neuron from the GPU.
   *            The GPU has the whole environment stored in its buffers; So data is available
   *            from when an iteration last calculated it. Because of this:
   *            !!WARNING!! Neuron data may not be up-to-date
   *
   * @param[in]   sequence_index    The sequence index to take the value from
   * @param[in]   past_index        The past index to take the value from
   * @param[in]   neuron_index      The relevant neuron index
   * @param[in]   data_set                The data set the network is evaluated on
   */
  double get_neuron_data(
    std::uint32_t sequence_index, std::uint32_t past_index, std::uint32_t neuron_index,
    const RafkoDataSet& data_set
  );

private:
  cl::Context m_openclContext;
  cl::Device m_openclDevice;
  cl::CommandQueue m_openclQueue;
  std::shared_ptr<AutoDiffGPUStrategy> m_strategy;
  rafko_mainframe::RafkoGPUPhase m_gpuPhase;
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OCL_OPTIMIZER_H */
