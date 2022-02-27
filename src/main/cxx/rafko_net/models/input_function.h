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

#ifndef RAFKO_INPUT_FUNCTION_H
#define RAFKO_INPUT_FUNCTION_H

#include "rafko_global.h"

#include <set>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net{

class RAFKO_FULL_EXPORT InputFunction{
public:
  inline static const std::set<Input_functions> all_input_functions = {
    input_function_add, input_function_multiply
  };

  static Input_functions next(std::set<Input_functions> range = {input_function_add});

  /**
   * @brief      Apply the given input function to the given inputs
   *
   * @param[in]  function   The function to apply
   * @param[in]  a          A value to merge through the input function
   * @param[in]  b          THe other value to merge with the input function
   *
   * @return     The result of data.
   */
  static constexpr double collect(Input_functions function, double a, double b){
    switch(function){
      case input_function_add: return a + b;
      case input_function_multiply: return a * b;
      case input_function_analog_xor: return std::abs(a - b) * (a + b);
      /*!Note: This solution for a number sequence of indefinite size might leave some mathematicians very furious, and rightly so.. '^^ */
      default: throw std::runtime_error("Unidentified Input function called!");
    };
  }

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   function    The Input function to base the generated kernel code on
   * @param[in]   a           A value to be merged through the input function
   * @param[in]   b           Another value to be merged through the input function
   *
   * @return    The generated Kernel code merging the parameters through the given input function
   */
  static std::string get_kernel_function_for(Input_functions function, std::string a, std::string b){
    switch(function){
      case input_function_add: return "( " + a + " + " + b + ")";
      case input_function_multiply: return  "( " + a + " * " + b + ")";
      case input_function_analog_xor: return "(abs(" + a + "-" + b +")" + " * (" + a + " + " + b + "))";
      default: throw std::runtime_error("Unidentified Input function called!");
    };
  }

  /**
   * @brief     Generates GPU kernel function code for the provided parameters
   *
   * @param[in]   operation_index   The variable containing a value from @get_kernel_enums
   * @param[in]   a                 A value to be merged through the input function
   * @param[in]   b                 Another value to be merged through the input function
   *
   * @return    The generated Kernel code merging the parameters through the given input function
   */
  static std::string get_kernel_function_for(std::string operation_index, std::string a, std::string b);

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    AN enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums(){
    return R"(
      typedef enum rafko_input_function_e{
        neuron_input_function_add = 0,
        neuron_input_function_multiply,
        neuron_input_function_analog_xor
      }rafko_input_function_t __attribute__ ((aligned));
    )";
  }
  #endif/*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_net */

#endif /* RAFKO_INPUT_FUNCTION_H */
