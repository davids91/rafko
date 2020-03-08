#ifndef WEIGHT_UPDATER_NESTEROV_H
#define WEIGHT_UPDATER_NESTEROV_H

#include "services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_nesterov : public Weight_updater{
public:
  Weight_updater_nesterov(SparseNet& sparse_net, Service_context service_context = Service_context())
  :  Weight_updater(sparse_net, service_context, 2)
  { }

protected: 
  sdouble32 get_new_weight(
    uint32 weight_index,
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
  ){
    if(is_finished()){
      return(
        net.weight_table(weight_index) - (
          (*previous_gradients[weight_index] * context.get_gamma())
          + (*gradients[weight_index] * context.get_step_size())
        )
      );
    }else{
      return(
        net.weight_table(weight_index) -( 
          ( /* Momentum based update */
            *gradients[weight_index]
            + (*previous_gradients[weight_index] * context.get_gamma())
          )
          * context.get_step_size()
        )
      );
    }
  }
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_NESTEROV_H */