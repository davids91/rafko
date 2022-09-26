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

#ifndef WEIGHT_UPDATER_AMSGRAD_H
#define WEIGHT_UPDATER_AMSGRAD_H

#include "rafko_gym/services/rafko_weight_updater.hpp"

namespace rafko_gym{

class RAFKO_EXPORT RafkoWeightUpdaterAMSGrad : public RafkoWeightUpdater{
public:
  RafkoWeightUpdaterAMSGrad(rafko_net::RafkoNet& rafko_net, const rafko_mainframe::RafkoSettings& settings)
  :  RafkoWeightUpdater(rafko_net, settings)
  ,  m_moment(rafko_net.weight_table_size(),(0.0))
  ,  m_rawMomentMax(rafko_net.weight_table_size(),(0.0))
  { }

  ~RafkoWeightUpdaterAMSGrad() = default;

  void iterate(const std::vector<double>& gradients) override;

protected:
  /**
   * @brief      Overridden member function from @RafkoWeightUpdater
   */
  double get_new_velocity(std::uint32_t weight_index, const std::vector<double>& /*gradients*/) const override{
    /*!Note: the variable moment contains the processed value of the gradients, so no need to use it here again. */
    return (
      m_settings.get_learning_rate() * m_moment[weight_index] / (
        std::sqrt(m_rawMomentMax[weight_index]) + m_settings.get_epsilon()
      )
    );
  }

private:
  std::uint32_t m_iterationCount = 0u;
  std::vector<double> m_moment;
  std::vector<double> m_rawMomentMax;
};

} /* namespace rafko_gym */

#endif /* WEIGHT_UPDATER_AMSGRAD_H */
