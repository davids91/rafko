#ifndef COST_FUNCTION_QUADRATIC_H
#define COST_FUNCTION_QUADRATIC_H

#include "models/cost_function.h"

#include <cmath>
#include <vector>

namespace sparse_net_library{

using std::vector;

static inline sdouble32 sample_distance_squared(sdouble32 feature_data, sdouble32 label_data){
  return pow((feature_data - label_data),2.0);
}

static inline sdouble32 sample_distance(sdouble32 feature_data, sdouble32 label_data){
  return (feature_data - label_data);
}

/**
 * @brief      Error function handling and utilities
 */
class Cost_function_quadratic : public Cost_function{
public:
  Cost_function_quadratic(vector<vector<sdouble32>>& label_samples, Service_context context = Service_context())
  : Cost_function(label_samples, context)
  { };

  sdouble32 get_error(vector<vector<sdouble32>>& features) const{
    verify_sizes(features);
    sdouble32 score = 0;
    uint32 feature_iterator = 0;
    uint32 feature_size = features[0].size();
    while(feature_iterator < feature_size){ /* evaluate a feature(Neuron output) on all samples */
      score += get_error(feature_iterator, features);
      ++feature_iterator;
    }
    return (0.5 * score)/features.size();
  }

  sdouble32 get_error(uint32 feature_index, vector<vector<sdouble32>>& features) const{
    verify_sizes(feature_index, features);
    return (0.5 * calculate_for_feature(feature_index, features, &sample_distance_squared))/features.size();
  }
  
  sdouble32 get_d_cost_over_d_feature(uint32 feature_index, vector<vector<sdouble32>>& features) const{
    verify_sizes(feature_index, features);
    return -2.0*(calculate_for_feature(feature_index, features, &sample_distance))/features.size();
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_QUADRATIC_H */
