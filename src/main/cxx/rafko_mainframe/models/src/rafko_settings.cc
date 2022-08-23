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

#include "rafko_mainframe/models/rafko_settings.hpp"


namespace rafko_mainframe{

double RafkoSettings::get_learning_rate(std::uint32_t iteration) const{
  if((0 == m_learningRateWithDecay.size())||(iteration < std::get<std::uint32_t>(m_learningRateWithDecay[0])))
    return m_hypers.learning_rate();
  if(iteration >= std::get<std::uint32_t>(m_learningRateWithDecay.back()))
    return std::get<double>(m_learningRateWithDecay.back());
  std::uint32_t decay_index = 0;
  if(iteration >= m_learningRateDecayIterationCache)
    decay_index = m_learningRateDecayIndexCache;

  while(
    (decay_index < (m_learningRateWithDecay.size()-1u))
    &&(iteration >= std::get<std::uint32_t>(m_learningRateWithDecay[decay_index]))
  )++decay_index;

  --decay_index;

  m_learningRateDecayIterationCache = iteration;
  m_learningRateDecayIndexCache = decay_index;

  return std::get<double>(m_learningRateWithDecay[decay_index]);
}

} /* rafko_mainframe */
