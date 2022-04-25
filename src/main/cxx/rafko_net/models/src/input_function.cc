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

#include "rafko_net/models/input_function.h"

#include <stdexcept>

#include "rafko_mainframe/services/rafko_assertion_logger.h"
#if(RAFKO_USES_OPENCL)
#include "rafko_utilities/services/rafko_string_utils.h"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_net {

Input_functions InputFunction::next(std::set<Input_functions> range){
  RFASSERT( 0u < range.size() );
  if(1u == range.size()) return *range.begin();

  Input_functions candidate = static_cast<Input_functions>(rand()%Input_functions_ARRAYSIZE);
  while(!Input_functions_IsValid(candidate)||find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<Input_functions>(rand()%Input_functions_ARRAYSIZE);

  return candidate;
}

double InputFunction::collect(Input_functions function, double a, double b){
  switch(function){
    case input_function_add: return a + b;
    case input_function_multiply: return a * b;
    /*!Note: This solution for a number sequence of indefinite size might leave some mathematicians very furious, and rightly so.. '^^ */
    default: throw std::runtime_error("Unidentified Input function called!");
  };
}

double InputFunction::get_derivative(Input_functions function, double a, double a_dw, double b, double b_dw){
  switch(function){
    case input_function_add: return a_dw + b_dw;
    case input_function_multiply: return (a * b_dw) + (a_dw * b);
    default: throw std::runtime_error("Unidentified Input function called!");
  };
}


#if(RAFKO_USES_OPENCL)
std::string InputFunction::get_kernel_function_for(std::string operation_index, std::string a, std::string b){
  std::string code = R"(
    switch(==op==){
      case neuron_input_function_add: ==a== = (==a== + ==b==); break;
      case neuron_input_function_multiply: ==a== = (==a== * ==b==); break;
      default: break;
    }
  )";
  code = rafko_utilities::replace_all_in_string(code, std::regex("==a=="), a);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==b=="), b);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="), operation_index);
  return code;
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
