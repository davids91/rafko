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

#include "rafko_net/models/input_function.hpp"

#include <stdexcept>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#if (RAFKO_USES_OPENCL)
#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif /*(RAFKO_USES_OPENCL)*/

namespace rafko_net {

Input_functions InputFunction::next(std::set<Input_functions> range) {
  RFASSERT(0u < range.size());
  if (1u == range.size())
    return *range.begin();

  Input_functions candidate =
      static_cast<Input_functions>(rand() % Input_functions_ARRAYSIZE);
  while (!Input_functions_IsValid(candidate) ||
         find(range.begin(), range.end(), candidate) == range.end())
    candidate =
        static_cast<Input_functions>(rand() % Input_functions_ARRAYSIZE);

  return candidate;
}

double InputFunction::collect(Input_functions function, double a, double b) {
  switch (function) {
  case input_function_add:
    return a + b;
  case input_function_multiply:
    return a * b;
  /*!Note: This solution for a number sequence of indefinite size might leave
   * some mathematicians very furious, and rightly so.. '^^ */
  default:
    throw std::runtime_error("Unidentified Input function called!");
  };
}

double InputFunction::get_derivative(Input_functions function, double a,
                                     double a_dw, double b, double b_dw) {
  switch (function) {
  case input_function_add:
    return a_dw + b_dw;
  case input_function_multiply:
    return (a * b_dw) + (a_dw * b);
  default:
    throw std::runtime_error("Unidentified Input function called!");
  };
}

#if (RAFKO_USES_OPENCL)
std::string
InputFunction::get_all_kernel_value_functions(std::string operation_index,
                                              std::string target, std::string a,
                                              std::string b) {
  std::string code = R"(
    switch(==op==){
      case input_function_add: ==target== = ((==a==) + (==b==)); break;
      case input_function_multiply: ==target== = ((==a==) * (==b==)); break;
      default: break;
    }
  )";
  code = rafko_utilities::replace_all_in_string(code, std::regex("==target=="),
                                                target);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==a=="), a);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==b=="), b);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="),
                                                operation_index);
  return code;
}

std::string InputFunction::get_all_kernel_derivative_functions(
    std::string operation_index, std::string target, std::string a,
    std::string a_dw, std::string b, std::string b_dw) {
  std::string code = R"(
    switch(==op==){
      case input_function_add: ==target== = ((==a_dw==) + (==b_dw==)); break;
      case input_function_multiply: ==target== = ((==a==) * (==b_dw==)) + ((==a_dw==) * (==b==)); break;
      default: break;
    }
  )";
  code = rafko_utilities::replace_all_in_string(code, std::regex("==target=="),
                                                target);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==a=="), a);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==b=="), b);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==a_dw=="),
                                                a_dw);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==b_dw=="),
                                                b_dw);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="),
                                                operation_index);
  return code;
}

std::string InputFunction::derivative_kernel_for(Input_functions function,
                                                 std::string a,
                                                 std::string a_dw,
                                                 std::string b,
                                                 std::string b_dw) {
  switch (function) {
  case input_function_add:
    return "((" + a_dw + ")+(" + b_dw + "))";
  case input_function_multiply:
    return "((" + a + ")*(" + b_dw + ")) + ((" + a_dw + ")*(" + b + "))";
  default:
    throw std::runtime_error("Unidentified Input function called!");
  };
}

#endif /*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
