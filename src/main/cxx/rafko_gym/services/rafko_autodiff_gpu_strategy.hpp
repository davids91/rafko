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

#ifndef RAFKO_AUTODIFF_GPU_STRATEGY_H
#define RAFKO_AUTODIFF_GPU_STRATEGY_H

#include "rafko_global.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"

#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/services/rafko_backpropagation_operation.hpp"
#include "rafko_gym/services/rafko_backprop_objective_operation.hpp"
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.hpp"
#include "rafko_gym/services/rafko_backprop_transfer_fn_operation.hpp"
#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.hpp"

namespace rafko_gym{

/**
 * @brief   A class implementing the underlying logic for an autodiff Backpropagation iteration step
 */
class AutoDiffGPUStrategy
: public rafko_mainframe::RafkoGPUStrategy
{
using OperationsType = std::shared_ptr<RafkoBackpropagationOperation>;
public:
  AutoDiffGPUStrategy(
    const rafko_mainframe::RafkoSettings& settings, rafko_net::RafkoNet& network,
    std::shared_ptr<RafkoDataSet> data_set = {}
  )
  : m_settings(settings)
  , m_network(network)
  {
    if(data_set)set_data_set(data_set);
  }

  void set_data_set(std::shared_ptr<RafkoDataSet> environment){
    m_dataSet = environment;
    RFASSERT(m_dataSet->get_input_size() == m_network.input_data_size());
    m_built = false;
  }

  cl::Program::Sources get_step_sources() const override{
    RFASSERT(m_built);
    RFASSERT(static_cast<bool>(m_dataSet));
    return {m_builtSource};
  }

  std::vector<std::string> get_step_names() const override{
    return {"autodiff_iterate"};
  }

  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const override;
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const override;

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const override{
    RFASSERT(static_cast<bool>(m_dataSet));
    std::size_t global_range = (
      std::min(m_settings.get_minibatch_size(), m_dataSet->get_number_of_sequences()) * m_maximumLocalWorkers
    );
    RFASSERT_LOG("Autodiff strategy global solution space: {}", global_range);
    return {cl::NullRange/*offset*/, cl::NDRange(global_range)/*global*/, cl::NDRange(m_maximumLocalWorkers)/*local*/ };
  }

  /**
   * @brief     Constructs the strategy based on the provided parameters
   *
   * @param[in]   operations                        The array of operations to process
   * @param[in]   weight_relevant_operation_count   The number of operations relevant to weights at the start of the opeartions array
   * 
   */
  void build(const std::vector<OperationsType>& operations, std::uint32_t weight_relevant_operation_count);

  static inline const std::uint32_t s_oneNeuralInstructionEntrySize = 6;
  const std::vector<std::uint32_t>& get_propagation_instructions() const{
    return m_neuralPropagationInstructions;
  }

private:
  const rafko_mainframe::RafkoSettings& m_settings;
  rafko_net::RafkoNet& m_network;
  std::shared_ptr<RafkoDataSet> m_dataSet;
  bool m_built = false;
  std::string m_builtSource;
  std::uint32_t m_numberOfOperations;
  std::uint32_t m_maximumLocalWorkers;
  std::vector<std::uint32_t> m_neuralPropagationInstructions;

  /**
   * @brief     Generates the instruction set to infer the Neural network on the GPU
   * 
   * @param[in]   operations          The array of operations to process
   *   
   * @return    An array to upload to GPU: the instruction index values representing the Neural network and parsed by the provided phase
   */
  [[nodiscard]] static std::vector<std::uint32_t> generate_propagation_instructions(
    const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>& operations
  );

  /**
   * @brief     Generates a 2D vector of operation index values
   *            where each operations in one row can be run in paralell, and each row depends on the previous one
   *            IMPORTANT: The function assumes that there are no cyclic dependencies
   *
   * @param[in]   operations    The array of operations to process
   *
   * @return    A 2D matrix of unsigned index values where each row can be run in paralell
   *            and the row ordering is ascending(i.e.: the last row depends on the previous rows)
   */
  [[nodiscard]] static std::vector<std::vector<std::uint32_t>> generate_operation_paralell_matrix(
    const std::vector<std::shared_ptr<RafkoBackpropagationOperation>>& operations
  );

  /**
   * @brief     Generates Kernel code to parse the Neural instruction information generated by @generate_propagation_instructions
   *            for forward propagation
   *
   * @param   network_input_array           The name of the arry containing the Inputs for the Neural network
   * @param   weight_array                  The name of the array contining the Neural network weights 
   * @param   operations_value_array        The name of the array containing the operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the operation values for both forward and backward propagation
   * @param   settings                      The settings instance to help in generating kernel parameters
   *
   * @return    The generated kernel code from the operations
   */
  static std::string generate_value_kernels(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size,
    const rafko_mainframe::RafkoSettings& settings
  );

  /**
   * @brief     Generates Kernel code to parse the Neural instruction information generated by @generate_propagation_instructions
   *            for backward propagation
   *
   * @param   network_input_array           The name of the arry containing the Inputs for the Neural network
   * @param   label_array                   The name of the arry containing the Labels the Neural network is evaluated against
   * @param   weight_array                  The name of the array contining the Neural network weights 
   * @param   operations_value_array        The name of the array containing the operation values for forward propagation
   * @param   operations_derivative_array   The name of the array containing the operation values for forward propagation
   * @param   operations_array_size         The size of the array contining the operation values for both forward and backward propagation
   * @param   settings                      The settings instance to help in generating kernel parameters
   *
   * @return    The generated kernel code from the operations
   */
  static std::string generate_derivative_kernels(
    std::string network_input_array, std::string label_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, const rafko_mainframe::RafkoSettings& settings
  );


  /**
   * @brief     Switches the burnt in placeholder values to actual values from the instance
   * 
   * @param[in]     kernel_source     The OpenCL Kernel to update
   */
  static void substitute_index_values_in_kernels(std::string& kernel_source);
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_GPU_STRATEGY_H */
