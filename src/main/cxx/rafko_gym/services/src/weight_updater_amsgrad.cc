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
#include "rafko_gym/services/weight_updater_amsgrad.hpp"

namespace rafko_gym{

void RafkoWeightUpdaterAMSGrad::iterate(const std::vector<double>& gradients){
  double raw_moment;
  for(std::uint32_t weight_index = 0; weight_index < m_moment.size(); ++weight_index){
    m_moment[weight_index] = (
      (m_settings.get_beta() * m_moment[weight_index])
      + ( (((1.0) - m_settings.get_beta()) * gradients[weight_index]) )
    );
    raw_moment = (
      (m_settings.get_beta_2() * m_rawMomentMax[weight_index])
      + (
        (((1.0) - m_settings.get_beta_2())
        * std::pow(gradients[weight_index], (2.0)))
      )
    );
    if(raw_moment > m_rawMomentMax[weight_index])
      m_rawMomentMax[weight_index] = raw_moment;
  }
  RafkoWeightUpdater::iterate(gradients);
  ++m_iterationCount;
}

} /* namespace rafko_gym */
