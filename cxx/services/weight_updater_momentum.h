#ifndef WEIGHT_UPDATER_MOMENTUM_H
#define WEIGHT_UPDATER_MOMENTUM_H

#include "services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_momentum : public Weight_updater{
public: 
  Weight_updater_momentum(SparseNet& sparse_net, Service_context service_context = Service_context())
  :  Weight_updater(sparse_net, service_context)
  { }

protected: 
  sdouble32 get_new_weight(
    uint32 weight_index,
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
  ){
    return(
      net.weight_table(weight_index) - ( 
        (*gradients[weight_index] * context.get_step_size())
        + (*previous_gradients[weight_index] * context.get_gamma() * context.get_step_size())
      )
    );
  }
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_MOMENTUM_H */