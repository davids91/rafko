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

#include "rafko_gym/services/cost_function.h"

#include <math.h>

namespace rafko_gym{

/**
 * @brief      Error function handling and utilities for Squared Error: C0 = ((y-y')^2)/2
 */
class RAFKO_FULL_EXPORT CostFunctionSquaredError : public CostFunction{
public:
  CostFunctionSquaredError(rafko_mainframe::RafkoSettings& settings)
  : CostFunction(cost_function_squared_error, settings)
  { };

protected:
  sdouble32 error_post_process(sdouble32 error_value, uint32 sample_number) const{
    parameter_not_used(sample_number);
    return error_value / double_literal(2.0);
  }

  mutable std::mutex cout_mutex;
  sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const{
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "{" + std::to_string(label_value) + "<>" + std::to_string(feature_value) + "} = "
    + std::to_string(pow((label_value - feature_value),2))
    << std::endl;
    return pow((label_value - feature_value),2);
  }

  #if(RAFKO_USES_OPENCL)
  std::string get_operation_kernel_source(std::string label_value, std::string feature_value) const{
    return "pow((" + label_value + " - " + feature_value + "),2.0)";
  }
  std::string get_post_process_kernel_source(std::string error_value) const{
    return "((" + error_value + ") / 2.0 )";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value, uint32 sample_number) const{
    parameter_not_used(sample_number);
    return -(label_value - feature_value);
  }
};

} /* namespace rafko_gym */

#endif /* COST_FUNCTION_SQUARED_ERROR_H */
