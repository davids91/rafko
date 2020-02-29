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

  sdouble32 get_error(uint32 sample_index, vector<sdouble32>& features) const{
    verify_sizes(sample_index, features);
    sdouble32 score = 0;
    uint32 feature_iterator = 0;
    while(feature_iterator < features.size()){ /* evaluate a feature(Neuron output) on the given sample */
      score += get_error(sample_index, feature_iterator, features[feature_iterator]);
      ++feature_iterator;
    }
    return score;
  }

  sdouble32 get_error(uint32 sample_index, uint32 label_index, sdouble32 feature_value) const{
    return ( 0.5 *
      pow((feature_value - labels[sample_index][label_index]),2) / static_cast<sdouble32>(labels[sample_index].size())
    );
  }

  sdouble32 get_d_cost_over_d_feature(uint32 sample_index, uint32 label_index, sdouble32 feature_value) const{
    return (
      -(feature_value - labels[sample_index][label_index]) / static_cast<sdouble32>(labels[sample_index].size())
    );
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_QUADRATIC_H */
