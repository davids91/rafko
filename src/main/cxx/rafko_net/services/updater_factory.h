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

#include "rafko_protocol/common.pb.h"

#include <memory>

#include "rafko_net/services/rafko_weight_updater.h"
#include "rafko_net/services/weight_updater_momentum.h"
#include "rafko_net/services/weight_updater_nesterovs.h"
#include "rafko_net/services/weight_updater_amsgrad.h"
#include "rafko_net/services/weight_updater_adam.h"

namespace rafko_net{

using std::unique_ptr;
using std::make_unique;

class UpdaterFactory{
public:
  /**
   * @brief      Builds a weight updater.
   *
   * @param      net             The network to base the weight updater upon
   * @param      solution        A reference of the Solution built from the network
   * @param[in]  weight_updater  The weight updater type
   * @param      context         The service context
   *
   * @return     The weight updater.
   */
  static unique_ptr<RafkoWeightUpdater> build_weight_updater(
    RafkoNet& net, Solution& solution, Weight_updaters weight_updater, RafkoServiceContext& context
  ){
    switch(weight_updater){
      case weight_updater_momentum:
        return make_unique<RafkoWeightUpdaterMomentum>(net, solution, context);
      case weight_updater_nesterovs:
        return make_unique<RafkoWeightUpdaterNesterovs>(net, solution, context);
      case weight_updater_adam:
        return make_unique<RafkoWeightUpdaterAdam>(net, solution, context);
      case weight_updater_amsgrad:
        return make_unique<RafkoWeightUpdaterAMSGrad>(net, solution, context);
      case weight_updater_default:
      default:
        return make_unique<RafkoWeightUpdater>(net, solution, context);
    };
  }
};

} /* namespace rafko_net */

#endif /* UPDATER_FACTORY_H */
