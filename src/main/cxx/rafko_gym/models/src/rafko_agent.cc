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

RafkoAgent::RafkoAgent(const rafko_net::Solution* solution, const rafko_mainframe::RafkoSettings& settings)
: m_settings(settings)
, m_solution(solution)
, m_maxThreadNumber(settings.get_max_processing_threads())
, m_commonDataPool()
, m_neuronValueBuffers(
  m_maxThreadNumber, rafko_utilities::DataRingbuffer<>(
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
{
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
