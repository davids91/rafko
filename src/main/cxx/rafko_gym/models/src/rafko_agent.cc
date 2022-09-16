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
#include "rafko_gym/models/rafko_agent.hpp"

namespace rafko_gym{

RafkoAgent::RafkoAgent(
  const rafko_net::Solution* solution, const rafko_mainframe::RafkoSettings& settings,
  std::uint32_t required_temp_data_size, std::uint32_t required_temp_data_number_per_thread,
  std::uint32_t max_threads
)
: m_settings(settings)
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

rafko_utilities::ConstVectorSubrange<> RafkoAgent::solve(
  const std::vector<double>& input, bool reset_neuron_data, std::uint32_t thread_index
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

#if(RAFKO_USES_OPENCL)
std::vector<rafko_mainframe::RafkoNBufShape> RafkoAgent::get_output_shapes() const{
  const std::size_t bytes_used = (
    std::max(m_sequencesEvaluating, 1u) /* number of sequences to evaluate */
    * std::max(2u, std::max( /* number of labels per sequence */
      m_solution->network_memory_length(), (m_sequenceSize + m_prefillInputsPerSequence)
    ) )
    * m_solution->neuron_number() /* number of numbers per label */
  );
  return{ rafko_mainframe::RafkoNBufShape{bytes_used, 1u} };
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_gym */
