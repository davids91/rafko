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

#include "rafko_global.hpp"

#include <set>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net{

class RAFKO_EXPORT InputFunction{
public:
  static inline const std::set<Input_functions> all_input_functions = {
    input_function_add, input_function_multiply
  };

  /**
   * @brief   Provides a random Input function based on the given range ( default is `input_function_add`)
   *
   * @param[in]   range   The range of input functions to pick the next one from
   */
  static Input_functions next(std::set<Input_functions> range = {input_function_add});

  /**
   * @brief      Apply the given input function to the given inputs
   *
   * @param[in]  function   The function to apply
   * @param[in]  a          A value to merge through the input function
   * @param[in]  b          THe other value to merge with the input function
   *
   * @return     The result of the collection.
   */
  static double collect(Input_functions function, double a, double b);

  /**
   * @brief      Calculate the derivative value of the given input function and the given inputs
   *
   * @param[in]  function   The function to apply
   * @param[in]  a          A value to merge through the input function
   * @param[in]  a_w        The derivative value of @a
   * @param[in]  b          The other value to merge with the input function
   * @param[in]  b_w        The derivative value of @b
   *
   * @return     The result of the provided input function's derivative
   */
  static double get_derivative(Input_functions function, double a, double a_dw, double b, double b_dw);

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
      default: throw std::runtime_error("Unidentified Input function called!");
    };
  }

  /**
   * @brief     Generates GPU code for the provided input function
   *
   * @param[in]   operation_index   The variable containing a value from @get_kernel_enums
   * @param[in]   a                 A value to be merged through the input function
   *                                The variable needs to hold a non-const reference in the kernel code so its value can be updated
   * @param[in]   b                 Another value to be merged through the input function
   *
   * @return    The generated Kernel code merging the parameters through the given input function
   */
  static std::string get_all_kernel_value_functions(std::string operation_index, std::string target, std::string a, std::string b);

  /**
   * @brief     Generates GPU code for all of the Input functions
   *
   * @param[in]   operation_index     The variable containing a value from @get_kernel_enums
   * @param[in]   a                   A value to merge through the input function
   *                                  The variable needs to hold a non-const reference in the kernel code so its value can be updated
   * @param[in]   a_w                 The derivative of @a
   * @param[in]   b                   The other value to merge with the input function
   * @param[in]   b_w                 The derivative of @b
   *
   * @return    The generated Kernel code merging the parameters through the given input function
   */
  static std::string get_all_kernel_derivative_functions(std::string operation_index, std::string target, std::string a, std::string a_dw, std::string b, std::string b_dw);

  /**
   * @brief      Provide the kernel code for derivative of the given input function and the given inputs
   *
   * @param[in]  function   The function to apply
   * @param[in]  a          A value to merge through the input function
   * @param[in]  a_w        The derivative value of @a
   * @param[in]  b          The other value to merge with the input function
   * @param[in]  b_w        The derivative value of @b
   *
   * @return     The single operation representing the derivative of the input function step
   */
  static std::string derivative_kernel_for(Input_functions function, std::string a, std::string a_dw, std::string b, std::string b_dw);


  /**
   * @brief     Gives back the identifier for the given function in the kernel
   *
   * @param[in]   function   The function to get the identifier to
   *
   * @return    The enumeration name for the given function
   */
  static std::string get_kernel_enum_for(Input_functions function){
    switch(function){
      case input_function_add: return "input_function_add";
      case input_function_multiply: return "input_function_multiply";
      default: throw std::runtime_error("Unidentified input function queried for information!");
    }
  }

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    AN enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums(){
    return R"(
      typedef enum rafko_input_function_e{
        input_function_unknown = 0,
        input_function_add,
        input_function_multiply
      }rafko_input_function_t __attribute__ ((aligned));
    )";
  }
  #endif/*(RAFKO_USES_OPENCL)*/
};

} /* namespace rafko_net */

#endif /* RAFKO_INPUT_FUNCTION_H */
