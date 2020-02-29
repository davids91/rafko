#ifndef WEIGHT_UPDATER_MOMENTUM_H
#define WEIGHT_UPDATER_MOMENTUM_H

#include "services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_momentum : public Weight_updater{
public: 
  Weight_updater_momentum(
    SparseNet& sparse_net,vector<unique_ptr<atomic<sdouble32>>>& weight_gradients_,
    Service_context& service_context
  ) : Weight_updater(sparse_net, weight_gradients_, service_context)
  { }
protected: 
  sdouble32 get_new_weight(uint32 weight_index){
    return(
      net.weight_table(weight_index) 
      + (*weight_gradients[weight_index] * context.get_step_size() * context.get_gamma())
    );
  }
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_MOMENTUM_H */