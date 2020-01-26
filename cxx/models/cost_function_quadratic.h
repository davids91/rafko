#ifndef COST_FUNCTION_QUADRATIC_H
#define COST_FUNCTION_QUADRATIC_H

#include "models/cost_function.h"

#include <cmath>
#include <vector>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      Error function handling and utilities for MSE: C0 = 1/2n(y-y')^2 */
class Cost_function_quadratic : public Cost_function{
public:
  Cost_function_quadratic(vector<vector<sdouble32>>& label_samples, Service_context context = Service_context())
  : Cost_function(label_samples, context)
  { };

  sdouble32 get_error(vector<vector<sdouble32>>& features) const{
    sdouble32 score = 0;
    uint32 sample_iterator = 0;
    for(vector<sdouble32>& feature_sample : features){ /* evaluate features(Neuron output) on all samples */
      if(feature_sample.size() != labels[sample_iterator].size())
        throw "Incompatible Feature and Label sizes!";
      score += get_error(sample_iterator, feature_sample);
      ++sample_iterator;
    }
    return score;
  }

  sdouble32 get_error(uint32 label_index, vector<sdouble32>& features) const{
    verify_sizes(label_index, features);
    sdouble32 score = 0;
    uint32 feature_iterator = 0;
    while(feature_iterator < features.size()){ /* evaluate a feature(Neuron output) on the given sample */
      score += get_error(label_index, feature_iterator, features);
      ++feature_iterator;
    }
    return score;
  }

  sdouble32 get_error(uint32 label_index, uint32 feature_index, vector<sdouble32>& features) const{
    return ( 0.5 * pow((features[feature_index] - labels[label_index][feature_index]),2) / static_cast<sdouble32>(features.size()) );
  }
  
  sdouble32 get_d_cost_over_d_feature(uint32 label_index, uint32 feature_index, vector<sdouble32>& features) const{
    return ( -(features[feature_index] - labels[label_index][feature_index]) / static_cast<sdouble32>(features.size()) );
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_QUADRATIC_H */
