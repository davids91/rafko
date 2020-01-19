#ifndef COST_FUNCTION_QUADRATIC_H
#define COST_FUNCTION_QUADRATIC_H

#include "models/cost_function.h"

#include <cmath>
#include <vector>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      Error function handling and utilities
 */
class Cost_function_quadratic : public Cost_function{
public:
  Cost_function_quadratic(
    vector<vector<sdouble32>>& feature_samples, vector<vector<sdouble32>>& label_samples, 
    Service_context context = Service_context()
  ) : Cost_function(feature_samples, label_samples, context){};

  sdouble32 get_error() const;
  sdouble32 get_error(uint32 feature_index) const;
  sdouble32 get_d_cost_over_d_feature(uint32 feature_index) const;
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_QUADRATIC_H */
