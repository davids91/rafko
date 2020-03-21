#ifndef WEIGHT_UPDATER_NESTEROV_H
#define WEIGHT_UPDATER_NESTEROV_H

#include "services/weight_updater.h"

namespace sparse_net_library{

class Weight_updater_nesterov : public Weight_updater{
public:
  Weight_updater_nesterov(SparseNet& sparse_net, Service_context service_context = Service_context())
  :  Weight_updater(sparse_net, service_context, 2)
  { }

  void iterate(vector<unique_ptr<atomic<sdouble32>>>& gradients,Solution& solution){
    Weight_updater::iterate(gradients, solution);
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity.begin());
  }

  void start(void){
    Weight_updater::start();
    std::copy(get_current_velocity().begin(),get_current_velocity().end(),previous_velocity_at_start.begin());
  }

protected: 
  sdouble32 get_new_velocity(uint32 weight_index, const vector<unique_ptr<atomic<sdouble32>>>& gradients){
    if(!is_finished()) return (
      (previous_velocity[weight_index] * context.get_gamma())
      + (*gradients[weight_index] * context.get_step_size())
    );
    else return(
      (previous_velocity_at_start[weight_index] * context.get_gamma())
      + (*gradients[weight_index] * context.get_step_size())
    );
  }

private:
  vector<sdouble32> previous_velocity_at_start;
  vector<sdouble32> previous_velocity;
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_NESTEROV_H */