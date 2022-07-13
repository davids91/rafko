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

#include "rafko_gym/services/cost_function.hpp"

#include <math.h>

namespace rafko_gym{

/**
 * @brief      Error function handling and utilities for MSE: C0 = 1/2n(y-y')^2
 */
class RAFKO_FULL_EXPORT CostFunctionMSE : public CostFunction{
public:
  CostFunctionMSE(const rafko_mainframe::RafkoSettings& settings)
  : CostFunction(cost_function_mse, settings)
  { };

protected:
  constexpr double error_post_process(double error_value, std::uint32_t sample_number) const override{
    return error_value / static_cast<double>(sample_number*(2.0));
  }

  constexpr double get_cell_error(double label_value, double feature_value) const override{
    return pow((label_value - feature_value),2);
  }

  constexpr double get_derivative(
    double label_value, double feature_value, double feature_d, double sample_number
  ) const override{
    return - sample_number * (label_value - feature_value) * feature_d;
  }

  #if(RAFKO_USES_OPENCL)
  std::string get_operation_kernel_source(std::string label_value, std::string feature_value) const override{
    return "pow((" + label_value + " - " + feature_value + "),2.0)";
  }

  std::string get_post_process_kernel_source(std::string error_value) const override{
    return "((" + error_value + ") / ((double)(sample_number) * 2.0) )";
  }

  std::string get_derivative_kernel_source(
    std::string label_value, std::string feature_value, std::string feature_d, std::string sample_number
  ) const override{
    return "(-" + sample_number + " * (" + label_value + " - " + feature_value + ") * " + feature_d + ")";
  }
  #endif/*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_gym */

#endif /* COST_FUNCTION_MSE_H */
