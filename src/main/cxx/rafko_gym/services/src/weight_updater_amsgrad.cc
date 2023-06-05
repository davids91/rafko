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

namespace rafko_gym {

void RafkoWeightUpdaterAMSGrad::iterate(const std::vector<double> &gradients) {
  for (std::int32_t weight_index = 0;
       weight_index < m_network.weight_table_size(); ++weight_index) {
    m_mean[weight_index] =
        ((m_settings.get_beta() * m_mean[weight_index]) +
         ((((1.0) - m_settings.get_beta()) * gradients[weight_index])));
    m_maxVariance[weight_index] =
        std::max(m_maxVariance[weight_index],
                 ((m_settings.get_beta_2() * m_maxVariance[weight_index]) +
                  ((((1.0) - m_settings.get_beta_2()) *
                    std::pow(gradients[weight_index], (2.0))))));
  }
  RafkoWeightUpdater::iterate(gradients);
  ++m_iterationCount;
}

double RafkoWeightUpdaterAMSGrad::get_new_velocity(
    std::uint32_t weight_index,
    const std::vector<double> & /*gradients*/) const {
  /*!Note: the variable moment contains the processed value of the gradients, so
   * no need to use it here again. */
  return -(
      (m_settings.get_learning_rate() /
       (std::sqrt(m_maxVariance[weight_index] /
                  ((1.0) - std::pow(m_settings.get_beta(),
                                    static_cast<double>(m_iterationCount)))) +
        m_settings.get_epsilon())) *
      (m_mean[weight_index] /
       ((1.0) - std::pow(m_settings.get_beta_2(),
                         static_cast<double>(m_iterationCount)))));
}

} /* namespace rafko_gym */
