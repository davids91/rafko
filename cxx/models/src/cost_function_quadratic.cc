#include "models/cost_function_quadratic.h"

namespace sparse_net_library{

static sdouble32 sample_distance_squared(sdouble32 feature_data, sdouble32 label_data){
  return pow((feature_data - label_data),2.0);
}

static sdouble32 sample_distance(sdouble32 feature_data, sdouble32 label_data){
  return (feature_data - label_data);
}

sdouble32 Cost_function_quadratic::get_error() const{
  verify_sizes();
  sdouble32 score = 0;
  uint32 feature_iterator = 0;
  uint32 feature_size = features[0].size();
  while(feature_iterator < feature_size){ /* evaluate a feature(Neuron output) on all samples */
    score += get_error(feature_iterator);
    ++feature_iterator;
  }
  return (0.5 * score)/features.size();
}

sdouble32 Cost_function_quadratic::get_error(uint32 feature_index) const{
  verify_sizes(feature_index);
  return (0.5 * calculate_for_feature(feature_index, &sample_distance_squared))/features.size();
}

sdouble32 Cost_function_quadratic::get_d_cost_over_d_feature(uint32 feature_index) const{
  verify_sizes(feature_index);
  return -2.0*(calculate_for_feature(feature_index, &sample_distance))/features.size();
}

} /* namespace sparse_net_library */
