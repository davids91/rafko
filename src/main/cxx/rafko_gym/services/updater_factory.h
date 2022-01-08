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

#ifndef UPDATER_FACTORY_H
#define UPDATER_FACTORY_H

#include "rafko_global.h"

#include <memory>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_gym/services/rafko_weight_updater.h"
#include "rafko_gym/services/weight_updater_momentum.h"
#include "rafko_gym/services/weight_updater_nesterovs.h"
#include "rafko_gym/services/weight_updater_amsgrad.h"
#include "rafko_gym/services/weight_updater_adam.h"

namespace rafko_gym{

class RAFKO_FULL_EXPORT UpdaterFactory{
public:
  /**
   * @brief      Builds a weight updater.
   *
   * @param      net             The network to base the weight updater upon
   * @param      solution        A reference of the Solution built from the network
   * @param[in]  weight_updater  The weight updater type
   * @param      settings         The service settings
   *
   * @return     The weight updater; Owner is arena, if settings has a pointer set, otherwise ownership is transferred to the caller of the function.
   */
  static std::unique_ptr<RafkoWeightUpdater> build_weight_updater(rafko_net::RafkoNet& net, rafko_net::Solution& solution, Weight_updaters weight_updater, rafko_mainframe::RafkoSettings& settings){
    switch(weight_updater){
      case weight_updater_momentum:   return std::make_unique<RafkoWeightUpdaterMomentum>(net, solution, settings);
      case weight_updater_nesterovs:  return std::make_unique<RafkoWeightUpdaterNesterovs>(net, solution, settings);
      case weight_updater_adam:       return std::make_unique<RafkoWeightUpdaterAdam>(net, solution, settings);
      case weight_updater_amsgrad:    return std::make_unique<RafkoWeightUpdaterAMSGrad>(net, solution, settings);
      case weight_updater_default:
      default:                        return std::make_unique<rafko_gym::RafkoWeightUpdater>(net, solution, settings);
    };
  }
};

} /* namespace rafko_gym */

#endif /* UPDATER_FACTORY_H */
