#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "sparse_net_global.h"

#include <vector>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      Error function handling and utilities
 */
class Cost_function{
public:
  Cost_function(uint8 maximum_threads, vector<vector<sdouble32>>& feature_samples, vector<vector<sdouble32>>& label_samples)
  : max_threads(maximum_threads), features(feature_samples), labels(label_samples){};

  virtual sdouble32 get_error() const = 0;
  virtual sdouble32 get_error(uint32 feature_index) const = 0;

protected:
  uint8 max_threads;
  vector<vector<sdouble32>>& features;
  vector<vector<sdouble32>>& labels;
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */
