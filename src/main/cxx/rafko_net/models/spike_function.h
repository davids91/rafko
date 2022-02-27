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

#include "rafko_global.h"

#include <vector>
#if(RAFKO_USES_OPENCL)
#include <string>
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.h"
#endif/*(RAFKO_USES_OPENCL)*/


namespace rafko_net{

/**
 * @brief      Spike function handling and utilities
 */
class RAFKO_FULL_EXPORT SpikeFunction{
public:
  /**
   * @brief      Apply the given spike function to a neurons activation data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to apply it to
   */
  static constexpr double get_value(double parameter, double new_data, double previous_data){
    return (previous_data * parameter) + (new_data * ((1.0)-parameter));
  }

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  static constexpr double get_derivative(double parameter, double transfer_function_output, double previous_value){
    parameter_not_used(parameter);
    return (previous_value - transfer_function_output);
  }

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   parameter       The Spike function to base the generated kernel code on
   * @param[in]   new_data        The result of the newly collected inputs and transfer function
   * @param[in]   previous_data   The previous activation value of the neuron
   *
   * @return    The generated Kernel code calling the asked spike function based on the parameter
   */
  static std::string get_cl_function_for(std::string parameter, std::string new_data, std::string previous_data){
    return "(((" + previous_data + ") * " + parameter + ") + ((" + new_data +") * (1.0 - " + parameter + ")))";
  }

  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   operation_index   The variable containing a value from @get_kernel_enums
   * @param[in]   parameter         The value of the input weight for the spike function
   * @param[in]   new_data          The value of the newly calculated neuron data
   * @param[in]   previous_data     The value of the previously present neuron data in which the result is stored,
   *                                So the variable needs to hold a reference in the kernel code
   *
   * @return    The generated Kernel code merging the parameters through the given input function
   */
  static std::string get_kernel_function_for(std::string operation_index, std::string parameter, std::string new_data, std::string previous_data){
    std::string code = R"(
      switch(==op==){
        case neuron_spike_function_none:
          ==previous_data== = ==new_data==;
          break;
        case neuron_spike_function_memory:
          ==previous_data== = (==previous_data== * ==parameter==) - (==new_data== * ==parameter==) + ==new_data==;
          break;
        case neuron_spike_function_p:
          ==previous_data== = ==previous_data== + ==parameter== * (==new_data== - ==previous_data==);
          break;
      }
    )";
    code = rafko_utilities::replace_all_in_string(code, std::regex("==parameter=="), parameter);
    code = rafko_utilities::replace_all_in_string(code, std::regex("==new_data=="), new_data);
    code = rafko_utilities::replace_all_in_string(code, std::regex("==previous_data=="), previous_data);
    code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="), operation_index);
    return code;
  }

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    AN enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums(){
    return R"(
      typedef enum rafko_spike_function_e{
        neuron_spike_function_none = 0,
        neuron_spike_function_memory,
        neuron_spike_function_p
      }rafko_spike_function_t __attribute__ ((aligned));
    )";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

};
} /* namespace rafko_net */
#endif /* SPIKE_FUNCTION_H */
