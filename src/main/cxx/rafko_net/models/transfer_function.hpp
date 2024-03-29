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

#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "rafko_global.hpp"

#include <set>
#if (RAFKO_USES_OPENCL)
#include <string>
#endif /*(RAFKO_USES_OPENCL)*/

#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_net {

/**
 * @brief      Transfer function handling and utilities
 */
class TransferFunction {
public:
  constexpr TransferFunction(const rafko_mainframe::RafkoSettings &settings)
      : m_settings(settings) {}

  /**
   * @brief      Gives a random Transfer Function
   *
   * @return     A random Transfer Function
   */
  static Transfer_functions next() {
    return next({transfer_function_identity, transfer_function_sigmoid,
                 transfer_function_tanh, transfer_function_elu,
                 transfer_function_selu, transfer_function_relu});
  }

  /**
   * @brief      Provides a random Transfer function out of the ones in the
   * argument
   *
   * @param[in]  range  The range of transfer functions to be given back
   *
   * @return     A random Transfer function according to the given range
   */
  static Transfer_functions next(std::set<Transfer_functions> range);

  /**
   * @brief      Provides the average range of the given Transfer functions
   * output
   *
   * @param[in]  function  The transfer function in question
   *
   * @return     The average output range.
   */
  static constexpr double
  get_average_output_range(Transfer_functions function) {
    switch (function) {
    case transfer_function_sigmoid:
    case transfer_function_tanh:
      return (1.0);
    case transfer_function_elu:
    case transfer_function_relu:
    case transfer_function_selu:
    case transfer_function_identity:
    default:
      return (50.0); /* The averagest number there is */
    }
  }

  /**
   * @brief      Apply the given transfer function to the given data
   *
   * @param[in]  function  The function to apply
   * @param[in]  data      The data to apply it to
   *
   * @return     The result of transfer_function(data).
   */
  double get_value(Transfer_functions function, double data) const;

  /**
   * @brief      Calculate the derivative of the given transfer function
   *
   * @param[in]  function   The function to apply
   * @param[in]  input      The input of the transfer function
   * @param[in]  input_dw   The derivative of the input of the transfer function
   *
   * @return     The derivative data
   */
  double get_derivative(Transfer_functions function, double input,
                        double input_dw) const;

#if (RAFKO_USES_OPENCL)

  /**
   * @brief     Generates GPU kernel function code for the value calculations
   * based on the provided parameters
   *
   * @param[in]   function    The Transfer function to base the generated kernel
   * code on
   * @param[in]   x           The value on which the transfer function is called
   * upon
   *
   * @return    The generated Kernel code of the transfer function
   */
  std::string get_kernel_function_for(Transfer_functions function,
                                      std::string x) const;

  /**
   * @brief     Generates GPU kernel function code for the derivative
   * calculations based on the provided parameters
   *
   * @param[in]  function   The function to apply
   * @param[in]  input      The input of the transfer function
   * @param[in]  input_dw   The derivative of the input of the transfer function
   *
   * @return    The generated Kernel code of the transfer function derivative
   */
  std::string get_kernel_function_for_d(Transfer_functions function,
                                        std::string input,
                                        std::string input_dw) const;

  /**
   * @brief     Generates GPU kernel code for the provided parameters and all
   * transfer functions
   *
   * @param[in]   transfer_function_index   The variable containing a value from
   * @get_kernel_enums
   * @param[in]   target  The target on which the transfer function result is
   * stored in
   * @param[in]   value   The value on which the transfer function is called
   * upon
   *
   * @return    The generated Kernel code containing all transfer functions,
   * selected by @transfer_function_index
   */
  static std::string
  get_all_kernel_value_functions(const rafko_mainframe::RafkoSettings &settings,
                                 std::string transfer_function_index,
                                 std::string target, std::string value);

  /**
   * @brief     Generates GPU kernel derivative function code for all transfer
   * functions
   *
   * @param[in]   transfer_function_index   The variable containing a value from
   * @get_kernel_enums
   *
   * @return    The generated Kernel code containing the derivative of all
   * transfer functions, selected by @transfer_function_index
   */
  static std::string get_all_kernel_derivative_functions(
      const rafko_mainframe::RafkoSettings &settings,
      std::string transfer_function_index, std::string target,
      std::string value, std::string derivative);

  /**
   * @brief     Gives back the identifier for the given function in the kernel
   *
   * @param[in]   function   The function to get the identifier to
   *
   * @return    The enumeration name for the given function
   */
  static std::string get_kernel_enum_for(Transfer_functions function) {
    switch (function) {
    case transfer_function_identity:
      return "transfer_function_identity";
    case transfer_function_sigmoid:
      return "transfer_function_sigmoid";
    case transfer_function_tanh:
      return "transfer_function_tanh";
    case transfer_function_elu:
      return "transfer_function_elu";
    case transfer_function_selu:
      return "transfer_function_selu";
    case transfer_function_relu:
      return "transfer_function_relu";
    case transfer_function_swish:
      return "transfer_function_swish";
    default:
      throw std::runtime_error(
          "Unidentified transfer function queried for information!");
    }
  }

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    An enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums() {
    return R"(
      typedef enum rafko_transfer_function_e{
        transfer_function_unknown = 0,
        transfer_function_identity,
        transfer_function_sigmoid,
        transfer_function_tanh,
        transfer_function_elu,
        transfer_function_selu,
        transfer_function_relu,
        transfer_function_swish
      }rafko_transfer_function_t __attribute__ ((aligned));
    )";
  }
#endif /*(RAFKO_USES_OPENCL)*/

private:
  const rafko_mainframe::RafkoSettings &m_settings;
};

} /* namespace rafko_net */
#endif /* TRANSFER_FUNCTION_H */
