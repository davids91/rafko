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

#include "gen/common.pb.h"

#include <memory>

#include "sparse_net_library/services/weight_updater.h"
#include "sparse_net_library/services/weight_updater_momentum.h"
#include "sparse_net_library/services/weight_updater_nesterov.h"
#include "sparse_net_library/services/weight_updater_amsgrad.h"
#include "sparse_net_library/services/weight_updater_adam.h"

namespace sparse_net_library{

using std::unique_ptr;
using std::make_unique;

class Updater_factory{
public:
  /**
   * @brief      Builds a weight updater.
   *
   * @param      net             The net to base the weight updater upon
   * @param[in]  weight_updater  The weight updater type
   * @param      context         The service context
   *
   * @return     The weight updater.
   */
  static unique_ptr<Weight_updater> build_weight_updater(
    SparseNet& net, weight_updaters weight_updater, Service_context& context
  ){
    switch(weight_updater){
      case WEIGHT_UPDATER_MOMENTUM:
        return make_unique<Weight_updater_momentum>(net,context);
      case WEIGHT_UPDATER_NESTEROV:
        return make_unique<Weight_updater_nesterov>(net,context);
      case WEIGHT_UPDATER_ADAM:
        return make_unique<Weight_updater_adam>(net,context);
      case WEIGHT_UPDATER_AMSGRAD:
        return make_unique<Weight_updater_amsgrad>(net,context);
      case WEIGHT_UPDATER_DEFAULT:
      default: 
        return make_unique<Weight_updater>(net,context);
    };
  }
};

} /* namespace sparse_net_library */

#endif /* UPDATER_FACTORY_H */
