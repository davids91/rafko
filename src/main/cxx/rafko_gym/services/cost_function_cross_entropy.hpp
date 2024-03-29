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

#ifndef COST_FUNCTION_CROSS_ENTROPY_H
#define COST_FUNCTION_CROSS_ENTROPY_H

#include "rafko_gym/services/cost_function.hpp"

#include <math.h>

namespace rafko_gym {

/**
 * @brief      Error function handling and utilities for Cross Entropy
 *             as described in
 * https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks
 */
class RAFKO_EXPORT CostFunctionCrossEntropy : public CostFunction {
public:
  CostFunctionCrossEntropy(const rafko_mainframe::RafkoSettings &settings)
      : CostFunction(cost_function_cross_entropy, settings){};
  ~CostFunctionCrossEntropy() = default;

#if (RAFKO_USES_OPENCL)
  /**
   * @brief      Provides the kernel function for the derivative of the cost
   * function
   *
   * @param[in]  label_value      The label value
   * @param[in]  feature_value    The data to comapre to the label value
   * @param[in]  feature_d        The derivative of the of the feature value
   * @param[in]  sample_number    The number of sample values the objective is
   * evaluated on at once
   *
   * @return     The source for implementing the kernel of the derivative of the
   * cost function
   */
  static std::string derivative_kernel_source(std::string label_value,
                                              std::string feature_value,
                                              std::string feature_d,
                                              std::string sample_number) {
    return "- (" + label_value + " * " + feature_d + ") / (" + sample_number +
           " * " + feature_value + ")";
  }
#endif /*(RAFKO_USES_OPENCL)*/

protected:
  [[nodiscard]] constexpr double
  error_post_process(double error_value,
                     std::uint32_t sample_number) const override {
    return (error_value / static_cast<double>(sample_number));
  }

  constexpr double get_cell_error(double label_value,
                                  double feature_value) const override {
    return (label_value *
            std::log(std::max(0.0000000000000001, feature_value)));
  }

  constexpr double get_derivative(double label_value, double feature_value,
                                  double feature_d,
                                  double sample_number) const override {
    return -(label_value * feature_d) / (sample_number * feature_value);
  }

#if (RAFKO_USES_OPENCL)
  std::string
  get_operation_kernel_source(std::string label_value,
                              std::string feature_value) const override {
    return "( " + label_value + " * log(max(0.0000000000000001," +
           feature_value + ")) )";
  }

  std::string
  get_post_process_kernel_source(std::string error_value) const override {
    return "((" + error_value + ") / (double)(sample_number) )";
  }

#endif /*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_gym */
#endif /* COST_FUNCTION_CROSS_ENTROPY_H */
