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

#ifndef WEIGHT_UPDATER_NESTEROVS_H
#define WEIGHT_UPDATER_NESTEROVS_H

#include "rafko_gym/services/rafko_weight_updater.hpp"

#include <vector>

namespace rafko_gym{

class RAFKO_FULL_EXPORT RafkoWeightUpdaterNesterovs : public RafkoWeightUpdater{
public:
  RafkoWeightUpdaterNesterovs(rafko_net::RafkoNet& rafko_net, const rafko_mainframe::RafkoSettings& settings)
  :  RafkoWeightUpdater(rafko_net, settings, 2u)
  { }

  void iterate(const std::vector<double>& gradients){
    RafkoWeightUpdater::iterate(gradients);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),m_previousVelocity.begin());
  }

  void start(){
    RafkoWeightUpdater::start();
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),m_previousVelocityAtStart.begin());
  }

protected:
  double get_new_velocity(std::uint32_t weight_index, const std::vector<double>& gradients) const override{
    if(!is_finished()) return (
      (m_previousVelocity[weight_index] * m_settings.get_gamma())
      + (gradients[weight_index] * m_settings.get_learning_rate())
    );
    else return(
      (m_previousVelocityAtStart[weight_index] * m_settings.get_gamma())
      + (gradients[weight_index] * m_settings.get_learning_rate())
    );
  }

private:
  std::vector<double> m_previousVelocityAtStart;
  std::vector<double> m_previousVelocity;
};

} /* namespace rafko_gym */

#endif /* WEIGHT_UPDATER_NESTEROVS_H */
