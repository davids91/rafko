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

#ifndef COST_FUNCTION_BINARY_CROSS_ENTROPY_H
#define COST_FUNCTION_BINARY_CROSS_ENTROPY_H

#include "rafko_gym/services/cost_function.h"

#include <math.h>

namespace rafko_gym{

/**
 * @brief      Error function handling and utilities for Binary Cross entropy sum
 *             as described in https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
 */
class RAFKO_FULL_EXPORT CostFunctionBinaryCrossEntropy : public CostFunction{
public:
  CostFunctionBinaryCrossEntropy(rafko_mainframe::RafkoSettings& settings)
  : CostFunction(cost_function_binary_cross_entropy, settings)
  { };

protected:
  sdouble32 error_post_process(sdouble32 error_value, uint32 sample_number) const{
    return ( error_value / static_cast<sdouble32>(sample_number) );
  }

  sdouble32 get_cell_error(sdouble32 label_value, sdouble32 feature_value) const{
    return (
      label_value * std::log(std::max(double_literal(0.0000000000000001), feature_value))
      +( (double_literal(1.0) - label_value) * std::log(double_literal(1.0) - std::min(double_literal(0.9999999999999999), feature_value)) )
    );
  }

  #if(RAFKO_USES_OPENCL)
  std::string get_operation_kernel_source(std::string label_value, std::string feature_value) const{
    std::string one_minus_label_value = "(1.0 - " + label_value + ")";
    return (
      "( " + label_value + " * log(max(0.0000000000000001," + feature_value + ")) )"
      + std::string("+") +
      "( " + one_minus_label_value + " * log(1.0 - min(0.9999999999999999," + feature_value + ")) )"
    );
  }
  std::string get_post_process_kernel_source(std::string error_value) const{
    return "((" + error_value + ") / double(sample_number) )";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  sdouble32 get_d_cost_over_d_feature(sdouble32 label_value, sdouble32 feature_value, uint32 sample_number) const{
    parameter_not_used(label_value);
    parameter_not_used(sample_number);
    return -( std::log(feature_value) + std::log((double_literal(1.0) - feature_value)) );
  }
};

} /* namespace rafko_gym */

#endif /* COST_FUNCTION_BINARY_CROSS_ENTROPY_H */
