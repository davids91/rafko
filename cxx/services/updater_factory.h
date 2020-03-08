#ifndef UPDATER_FACTORY_H
#define UPDATER_FACTORY_H

#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "models/service_context.h"
#include "services/weight_updater.h"
#include "services/weight_updater_momentum.h"
#include "services/weight_updater_nesterov.h"

#include <memory>

namespace sparse_net_library{

using std::unique_ptr;
using std::make_unique;

class Updater_factory{
public:
  static unique_ptr<Weight_updater> build_weight_updater(
    SparseNet& net, weight_updaters weight_updater, Service_context& context
  ){
    switch(weight_updater){
      case WEIGHT_UPDATER_MOMENTUM: 
        return make_unique<Weight_updater_momentum>(net,context);
      case WEIGHT_UPDATER_NESTEROV:
        return make_unique<Weight_updater_nesterov>(net,context);
      case WEIGHT_UPDATER_DEFAULT: 
      default: 
        return make_unique<Weight_updater>(net,context);
    };
  }
};

} /* namespace sparse_net_library */

#endif /* UPDATER_FACTORY_H */
