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

#ifndef RAFKO_AGENT_H
#define RAFKO_AGENT_H

#include "rafko_global.hpp"

#include <vector>
#include <functional>
#if(RAFKO_USES_OPENCL)
#include <numeric>
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#include "rafko_utilities/models/data_pool.hpp"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#if(RAFKO_USES_OPENCL)
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.hpp"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_gym{

/**
 * @brief      This class serves as a base for reinforcement learning agent, which provides output data
 *              based on different inputs
 */
class RAFKO_EXPORT RafkoAgent
#if(RAFKO_USES_OPENCL)
: public rafko_mainframe::RafkoGPUStrategyPhase
#endif/*(RAFKO_USES_OPENCL)*/
{
public:
  RafkoAgent(const rafko_net::Solution* solution, const rafko_mainframe::RafkoSettings& settings);
  virtual ~RafkoAgent() = default;

  /*
   * @brief   Sets evaluation mode for agent, which signals whether or not training relevant Neural features(e.g. dropout) are to be executed.
   *
   * @param[in]   evaluation    decides whether the agent is in evaluation mode or not
   *
   */
  virtual void set_eval_mode(bool evaluation) = 0;

  /**
   * @brief      For the provided input, return the result of the neural network
   *
   * @param[in]      input                  The input data to be taken
   * @param[in]      reset_neuron_data      should the internal memory of the solver is to be resetted before solving the neural network
   * @param[in]      thread_index           The index of thread the solution is to be running from
   *
   * @return         The output values of the network result
   */
  virtual rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input, bool reset_neuron_data = false, std::uint32_t thread_index = 0u
  ) = 0;


  /**
   * @brief      Provide the raw Neural data
   *
   * @param[in]      thread_index     The index of the target thread
   * @return         A const reference to the raw Neuron data
   */
  const rafko_utilities::DataRingbuffer<>& get_memory(std::uint32_t thread_index = 0) const{
    RFASSERT(thread_index < m_neuronValueBuffers.size());
    return m_neuronValueBuffers[thread_index];
  }

#if(RAFKO_USES_OPENCL)
  /**
   * @brief     Sets the parameters the generated kernel code will be based on. These parameters usually come from an environment
   *
   * @param[in]     sequence_number                 How many sequences are there?
   * @param[in]     sequence_size                   How many feature-label pairs are in a sequence?
   * @param[in]     prefill_inputs_per_sequence     How many prefill inputs are there before a sequence?
   */
  constexpr void set_sequence_params(
    std::uint32_t sequence_number, std::uint32_t sequence_size = 1u, std::uint32_t prefill_inputs_per_sequence = 0u
  ){
    m_sequencesEvaluating = sequence_number;
    m_sequenceSize = sequence_size;
    m_prefillInputsPerSequence = prefill_inputs_per_sequence;
  }

  cl::Program::Sources get_step_sources() const override{
    return { rafko_net::SolutionBuilder::get_kernel_for_solution(
      *m_solution, "agent_solution",
      m_sequenceSize, m_prefillInputsPerSequence,
      m_settings
    ) };
  }

  std::vector<std::string> get_step_names() const override{
    return {"agent_solution"};
  }

  /**
   * @brief      Provides the input dimension of the agent, which consist of
   *             3 buffers: mode, weights, and (inputs + prefill) for each evaluated sequence
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const override{
    return{ rafko_mainframe::RafkoNBufShape{
      1u, m_deviceWeightTableSize,
      (m_sequencesEvaluating * (m_sequenceSize + m_prefillInputsPerSequence) * m_solution->network_input_size())
    } };
  }

  /**
   * @brief      Provides the output dimension of the agent, which consist of
   *             1 buffer: Neuron outputs for each evaluated sequence or network memory
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   *             Agent output structure: {{ used bytes for execution, used bytes for performance feature error summary }}
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const override;

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const override{
    return {cl::NullRange,cl::NDRange(m_sequencesEvaluating),cl::NullRange};
  }

#endif/*(RAFKO_USES_OPENCL)*/

protected:
  const rafko_mainframe::RafkoSettings& m_settings;
  const rafko_net::Solution* m_solution;
  std::uint32_t m_maxThreadNumber;
  rafko_utilities::DataPool<double> m_commonDataPool;
  std::vector<rafko_utilities::DataRingbuffer<>> m_neuronValueBuffers; /* One rafko_utilities::DataRingbuffer per thread */
  std::vector<std::reference_wrapper<std::vector<double>>> m_usedDataBuffers;

private:
#if(RAFKO_USES_OPENCL)
  std::uint32_t m_sequencesEvaluating = 1u;
  std::uint32_t m_sequenceSize = 1u;
  std::uint32_t m_prefillInputsPerSequence = 0u;
  std::uint32_t m_deviceWeightTableSize;
#endif/*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_gym */
#endif /* RAFKO_AGENT_H */
