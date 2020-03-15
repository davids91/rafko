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

#ifndef COST_FUNCTION_SQUARED_ERROR_H
#define COST_FUNCTION_SQUARED_ERROR_H

#include "models/cost_function.h"

#include <cmath>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      Error function handling and utilities for Squared Error: C0 = ((y-y')^2)/2 */
class Cost_function_squared_error : public Cost_function{
public:
  Cost_function_squared_error(uint32 feature_size_, Service_context service_context = Service_context())
  : Cost_function(feature_size_, service_context)
  { };

protected:
  sdouble32 error_post_process(sdouble32 error_value) const{
    return error_value / double_literal(2.0);
  }

  sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const{
    return pow((label_value - feature_value),2);
  }

  sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value) const{
    return -(label_value - feature_value);
  }

private:
  sdouble32 sample_number;
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_SQUARED_ERROR_H */
