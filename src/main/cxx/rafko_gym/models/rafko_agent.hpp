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
  RafkoAgent(
    const rafko_net::Solution* solution, const rafko_mainframe::RafkoSettings& settings,
    std::uint32_t required_temp_data_size, std::uint32_t required_temp_data_number_per_thread,
    std::uint32_t max_threads = 1u
  ):m_settings(settings)
  , m_solution(solution)
  , m_requiredTempDataNumberPerThread(required_temp_data_number_per_thread)
  , m_requiredTempDataSize(required_temp_data_size)
  , m_maxThreads(max_threads)
  , m_commonDataPool((m_requiredTempDataNumberPerThread * m_maxThreads), m_requiredTempDataSize)
  , m_neuronValueBuffers(
    m_maxThreads, rafko_utilities::DataRingbuffer<>(
      m_solution->network_memory_length(),
      [this](std::vector<double>& buffer){
        buffer = std::vector<double>(m_solution->neuron_number(), 0.0);
      }
    )
  )
  #if(RAFKO_USES_OPENCL)
  , m_deviceWeightTableSize( std::accumulate(
    m_solution->partial_solutions().begin(), m_solution->partial_solutions().end(), 0u,
    [](const std::uint32_t& sum, const rafko_net::PartialSolution& partial){
      return ( sum + partial.weight_table_size() );
    }
  ) )
  #endif/*(RAFKO_USES_OPENCL)*/
  { /* A temporary buffer is allocated for every required future usage per thread */
    for(std::uint32_t tmp_data_index = 0; tmp_data_index < (m_requiredTempDataNumberPerThread * m_maxThreads); ++tmp_data_index)
      m_usedDataBuffers.push_back(m_commonDataPool.reserve_buffer(m_requiredTempDataSize));
  }

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
  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input, bool reset_neuron_data = false, std::uint32_t thread_index = 0u
  ){
    if(m_maxThreads > thread_index){
      if( input.size() != m_solution->network_input_size() )
        throw std::runtime_error(
          "Input size(" + std::to_string(input.size()) + ") doesn't match "
          + std::string("networks input size(") + std::to_string(m_solution->network_input_size()) + ")!"
        );

      if(reset_neuron_data)m_neuronValueBuffers[thread_index].reset();
      solve( input, m_neuronValueBuffers[thread_index], m_usedDataBuffers, (thread_index * m_requiredTempDataNumberPerThread), thread_index );
      return { /* return with the range of the output Neurons */
        m_neuronValueBuffers[thread_index].get_element(0).end() - m_solution->output_neuron_number(),
        m_neuronValueBuffers[thread_index].get_element(0).end()
      };
    } else throw std::runtime_error("Thread index out of bounds!");
  }

  /**
   * @brief      Solves the rafko_net::Solution provided in the constructor, previous neural information is supposedly available in @output buffer
   *
   * @param[in]      input                    The input data to be taken
   * @param          output                   The used Output data to write the results to
   * @param[in]      tmp_data_pool            The already allocated data pool to be used to store intermediate data
   * @param[in]      used_data_pool_start     The first index inside @tmp_data_pool to be used
   */
  virtual void solve(
    const std::vector<double>& input, rafko_utilities::DataRingbuffer<>& output,
    const std::vector<std::reference_wrapper<std::vector<double>>>& tmp_data_pool,
    std::uint32_t used_data_pool_start = 0, std::uint32_t thread_index = 0
  ) const = 0;

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

  virtual ~RafkoAgent() = default;

  /**
   * @brief     Provides the size of the buffer it was declared with
   */
  constexpr std::uint32_t get_required_temp_data_size() const{
    return m_requiredTempDataSize;
  }

#if(RAFKO_USES_OPENCL)
  constexpr void set_sequence_params(std::uint32_t sequence_number, std::uint32_t sequence_size = 1u, std::uint32_t prefill_inputs_per_sequence = 0u){
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
  std::vector<std::string> get_step_names() const{
    return {"agent_solution"};
  }

  /**
   * @brief      Provides the input dimension of the agent, which consist of
   *             3 buffers: mode, weights, and (inputs + prefill) for each evaluated sequence
   *
   * @return     Vector of dimensions in order of @get_step_sources and @get_step_names
   */
  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const{
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
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const{
    const std::size_t bytes_used = (
      std::max(m_sequencesEvaluating, 1u) /* number of sequences to evaluate */
      * std::max(2u, std::max( /* number of labels per sequence */
        m_solution->network_memory_length(), (m_sequenceSize + m_prefillInputsPerSequence)
      ) )
      * m_solution->neuron_number() /* number of numbers per label */
    );
    return{ rafko_mainframe::RafkoNBufShape{bytes_used, 1u} };
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const{
    return std::make_tuple( cl::NullRange/*offset*/, cl::NDRange(m_sequencesEvaluating)/*global*/, cl::NullRange/*local*/ );
  }
#endif/*(RAFKO_USES_OPENCL)*/

protected:
  const rafko_mainframe::RafkoSettings& m_settings;
  const rafko_net::Solution* m_solution;
  std::uint32_t m_requiredTempDataNumberPerThread;
  std::uint32_t m_requiredTempDataSize;
  std::uint32_t m_maxThreads;

private:
  mutable rafko_utilities::DataPool<double> m_commonDataPool;
  std::vector<rafko_utilities::DataRingbuffer<>> m_neuronValueBuffers; /* One rafko_utilities::DataRingbuffer per thread */
  std::vector<std::reference_wrapper<std::vector<double>>> m_usedDataBuffers;
#if(RAFKO_USES_OPENCL)
  std::uint32_t m_sequencesEvaluating = 1u;
  std::uint32_t m_sequenceSize = 1u;
  std::uint32_t m_prefillInputsPerSequence = 0u;
  std::uint32_t m_deviceWeightTableSize;
#endif/*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_gym */
#endif /* RAFKO_AGENT_H */
