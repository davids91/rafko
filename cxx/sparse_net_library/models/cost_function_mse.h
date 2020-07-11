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

#ifndef COST_FUNCTION_MSE_H
#define COST_FUNCTION_MSE_H

#include "sparse_net_library/models/cost_function.h"

#include <cmath>

namespace sparse_net_library{

using std::vector;

using rafko_mainframe::Service_context;

/**
 * @brief      Error function handling and utilities for MSE: C0 = 1/2n(y-y')^2 */
class Cost_function_mse : public Cost_function{
public:
  Cost_function_mse(uint32 feature_size_, uint32 sample_number_, Service_context service_context = Service_context())
  : Cost_function(feature_size_, service_context)
  , sample_number(sample_number_)
  { };

protected:
  sdouble32 error_post_process(sdouble32 error_value) const{
    return error_value / static_cast<sdouble32>(sample_number*double_literal(2.0));
  }

  sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const{
    return pow((label_value - feature_value),2);
  }

  sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value) const{
    return -(label_value - feature_value) / static_cast<sdouble32>(sample_number);
  }

private:
  sdouble32 sample_number;
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_MSE_H */
