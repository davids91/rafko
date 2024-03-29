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

#ifndef SPIKE_FUNCTION_H
#define SPIKE_FUNCTION_H

#include "rafko_global.hpp"

#include "rafko_protocol/rafko_net.pb.h"

#include <set>
#if (RAFKO_USES_OPENCL)
#include <string>
#endif /*(RAFKO_USES_OPENCL)*/

namespace rafko_net {

/**
 * @brief      Spike function handling and utilities
 */
class RAFKO_EXPORT SpikeFunction {
public:
  static inline const std::set<Spike_functions> all_spike_functions = {
      spike_function_none, spike_function_memory, spike_function_p,
      spike_function_amplify_value};

  /**
   * @brief   Provides a random Input function based on the given range (
   * default is `input_function_add`)
   *
   * @param[in]   range   The range of input functions to pick the next one from
   */
  static Spike_functions next(std::set<Spike_functions> range = {
                                  spike_function_memory});

  /**
   * @brief      Apply the given spike function to a neurons activation data
   *
   * @param[in]   function        The function to apply
   * @param[in]   parameter       The parameter supplied by a Neuron
   * @param[in]   new_data        The latest data as input to the spike function
   * @param[in]   previous_data   The previously stored state of the Spike
   * function
   */
  static double get_value(Spike_functions function, double parameter,
                          double new_data, double previous_data);

  /**
   * @brief      Calculates the derivative of the spike function
   *             in case the basis of the derivative is the relevant parameter
   *
   * @param[in]   function          The function to apply
   * @param[in]   parameter         The parameter of the spike function
   * @param[in]   previous_data     The previously stored state of the Spike
   * function
   * @param[in]   previous_data_d   The derivative of the previously stored
   * state
   * @param[in]   new_data          The latest data as input to the spike
   * function
   * @param[in]   new_data_d        The derivative of the latest data
   */
  static double get_derivative_for_w(Spike_functions function, double parameter,
                                     double previous_data,
                                     double previous_data_d, double new_data,
                                     double new_data_d);

  /**
   * @brief      Calculates the derivative of the spike function
   *             in case the basis of the derivative is not the relevant
   * parameter
   *
   * @param[in]   function          The function to apply
   * @param[in]   parameter         The parameter of the spike function
   * @param[in]   previous_data_d   The derivative of the previously stored
   * @param[in]   new_data_d        The derivative of the latest data
   * state
   */
  static double get_derivative_not_for_w(Spike_functions function,
                                         double parameter,
                                         double previous_data_d,
                                         double new_data_d);

#if (RAFKO_USES_OPENCL)
  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   function        The function to apply
   * @param[in]   parameter       The Spike function to base the generated
   * kernel code on
   * @param[in]   previous_data   The previous activation value of the neuron
   * @param[in]   new_data        The result of the newly collected inputs and
   * transfer function
   *
   * @return    The generated Kernel code calling the asked spike function based
   * on the parameter
   */
  static std::string get_kernel_function_for(Spike_functions function,
                                             std::string parameter,
                                             std::string previous_data,
                                             std::string new_data);

  /**
   * @brief     Generates GPU code for the provided spike function and
   * parameters
   *
   * @param[in]   spike_fn_index   The variable containing a value from
   * @get_kernel_enums
   * @param[in]   target            The target on which to store the results
   * @param[in]   parameter         The value of the input weight for the spike
   * function
   * @param[in]   previous_data     The value of the previously present neuron
   * data in which the result is stored.
   * @param[in]   new_data          The value of the newly calculated neuron
   * data
   *
   * @return    The generated Kernel code merging the parameters through the
   * given input function
   */
  static std::string get_all_kernel_value_functions(std::string spike_fn_index,
                                                    std::string target,
                                                    std::string parameter,
                                                    std::string previous_data,
                                                    std::string new_data);

  /**
   * @brief     Generates GPU code for all of the spike function derivatives in
   * case the derivative base weight index matches the one used in the spike
   * function
   *
   * @param[in]   spike_fn_index   The variable containing a value from
   * @get_kernel_enums
   * @param[in]   target            The target on which to store the results
   * @param[in]   parameter         The value of the input weight for the spike
   * function
   * @param[in]   previous_data     The previously stored state of the Spike
   * function
   * @param[in]   previous_data_d   The derivative of the previously stored
   * state
   * @param[in]   new_data          The latest data as input to the spike
   * function
   * @param[in]   new_data_d        The derivative of the latest data
   *
   * @return    The generated Kernel code containing all of the Spike functions,
   * the one being executed selected by @spike_fn_index
   */
  static std::string get_all_kernel_derivative_functions_for_w(
      std::string spike_fn_index, std::string target, std::string parameter,
      std::string previous_data, std::string previous_data_d,
      std::string new_data, std::string new_data_d);

  /**
   * @brief     Generates GPU code for all of the spike function derivatives in
   * case the derivative base weight index doesn't match the one used in the
   * spike function
   *
   * @param[in]   spike_fn_index   The variable containing a value from
   * @get_kernel_enums
   * @param[in]   target            The target on which to store the results
   * @param[in]   parameter         The value of the input weight for the spike
   * function
   * @param[in]   previous_data_d   The derivative of the previously stored
   * state
   * @param[in]   new_data          The latest data as input to the spike
   * function
   * @param[in]   new_data_d        The derivative of the latest data
   *
   * @return    The generated Kernel code containing all of the Spike functions,
   * the one being executed selected by @spike_fn_index
   */
  static std::string get_all_kernel_derivative_functions_not_for_w(
      std::string spike_fn_index, std::string target, std::string parameter,
      std::string previous_data_d, std::string new_data_d);

  /**
   * @brief      Provides the derivative kernel for the derivative of the spike
   * function in case the basis of the derivative is the relevant parameter
   *
   * @param[in]   function          The function to apply
   * @param[in]   parameter         The parameter of the spike function
   * @param[in]   new_data          The latest data as input to the spike
   * function
   * @param[in]   new_data_d        The derivative of the latest data
   * @param[in]   previous_data     The previously stored state of the Spike
   * function
   * @param[in]   previous_data_d   The derivative of the previously stored
   * state
   *
   * @return    The single kernel operation representing the provided spike
   * functions derivative function
   */
  static std::string get_derivative_kernel_for_w(Spike_functions function,
                                                 std::string parameter,
                                                 std::string previous_data,
                                                 std::string previous_data_d,
                                                 std::string new_data,
                                                 std::string new_data_d);

  /**
   * @brief      Provides the derivative kernel for the derivative of the spike
   * function in case the basis of the derivative is not the relevant parameter
   *
   * @param[in]   function          The function to apply
   * @param[in]   parameter         The parameter of the spike function
   * @param[in]   new_data_d        The derivative of the latest data
   * @param[in]   previous_data_d   The derivative of the previously stored
   * state
   *
   * @return    The single kernel operation representing the provided spike
   * functions derivative function
   */
  static std::string get_derivative_kernel_not_for_w(
      Spike_functions function, std::string parameter,
      std::string previous_data_d, std::string new_data_d);

  /**
   * @brief     Gives back the identifier for the given function in the kernel
   *
   * @param[in]   function   The function to get the identifier to
   *
   * @return    The enumeration name for the given function
   */
  static std::string get_kernel_enum_for(Spike_functions function);

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    An enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums() {
    return R"(
      typedef enum rafko_spike_function_e{
        spike_function_unknown = 0,
        spike_function_none,
        spike_function_memory,
        spike_function_p,
        spike_function_amplify_value
      }rafko_spike_function_t __attribute__ ((aligned));
    )";
  }
#endif /*(RAFKO_USES_OPENCL)*/
};
} /* namespace rafko_net */
#endif /* SPIKE_FUNCTION_H */
