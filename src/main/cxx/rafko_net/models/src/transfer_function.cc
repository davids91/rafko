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

#include "rafko_net/models/transfer_function.h"

#include <math.h>
#include <stdexcept>
#if(RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.h"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_net {

Transfer_functions TransferFunction::next(std::set<Transfer_functions> range){
  RFASSERT( 0u < range.size() );
  if(1u == range.size()) return *range.begin();

  Transfer_functions candidate = static_cast<Transfer_functions>(rand()%Transfer_functions_ARRAYSIZE);
  while(!Transfer_functions_IsValid(candidate)||find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<Transfer_functions>(rand()%Transfer_functions_ARRAYSIZE);

  return candidate;
}

double TransferFunction::get_value(Transfer_functions function, double data) const{
  switch(function){
    case transfer_function_identity: return data; /* Identity means f(x) = x */
    case transfer_function_sigmoid: return 1.0/(1.0+std::exp(-data));
    case transfer_function_tanh: return std::tanh(data);
    case transfer_function_elu:
      if(data <= 0.0) return settings.get_alpha() * (std::exp(data) - 1.0);
      else return data;
    case transfer_function_selu:
      if(data <= 0.0) return settings.get_lambda() * settings.get_alpha() * (std::exp(data) - 1.0);
      else return settings.get_lambda() * data;
    case transfer_function_relu: return std::max((0.0),data);
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

double TransferFunction::get_derivative(Transfer_functions function, double input, double input_dw) const{
  switch(function){
  case transfer_function_identity: return input_dw;
  case transfer_function_sigmoid:
    return (input_dw * std::exp(input))/std::pow((std::exp(input) + 1.0),2.0);
  case transfer_function_tanh: input_dw / std::pow(std::cosh(input), 2.0);
  case transfer_function_elu:
    if(input <= 0.0) return settings.get_alpha() * std::exp(input) * input_dw;
    else return input_dw;
  case transfer_function_selu:
    if(input <= 0.0) return settings.get_lambda() * settings.get_alpha() * std::exp(input) * input_dw;
    else return settings.get_lambda() * input_dw;
  case transfer_function_relu:
    if(input <= 0.0) return 0;
    else return input_dw;
  }
}

#if(RAFKO_USES_OPENCL)
std::string TransferFunction::get_cl_function_for(Transfer_functions function, std::string x_){
  std::string x = std::string("(") + x_ + ")";
  switch(function){
    case transfer_function_identity: return x;
    case transfer_function_sigmoid: return "( 1.0/(1.0 + exp( -" + x + ")) )";
    case transfer_function_tanh: return "(tanh(" + x + "))";
    case transfer_function_elu: return "( max(0.0," + x + ") + (" + std::to_string(settings.get_alpha()) + " * (exp(min(0.0, " + x + ")) - 1.0)) )";
    case transfer_function_selu:
    {
      std::string alpha = std::to_string(settings.get_alpha());
      std::string lambda = std::to_string(settings.get_lambda());
      std::string x_negative_component = "min(0.0, " + x + ")";
      std::string x_positive_component = "max(0.0, " + x + ")";
      std::string x_negative_scaled = "(" + alpha + " * (exp(" + x_negative_component + ") - 1.0) )";
      return "( " + lambda + " * (" + x_positive_component + " + " + x_negative_scaled + ") )";
    }
    case transfer_function_relu: return "max(0.0," + x + ")";
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

std::string TransferFunction::get_kernel_function_for(std::string operation_index, std::string a, std::string b){
  std::string code = R"(
    switch(==op==){
      case neuron_transfer_function_identity:
        ==a== = ==b==;
        break;
      case neuron_transfer_function_sigmoid:
        ==a== = 1.0/(1.0+exp(-==b==));
        break;
      case neuron_transfer_function_tanh:
        ==a== = tanh(==b==);
        break;
      case neuron_transfer_function_elu:
        ==a== = (
          max(0.0, ==b==) + ( ==alpha== * (exp(min(0.0, ==b==)) - 1.0) )
        );
        break;
      case neuron_transfer_function_selu:
        ==a== = ==lambda== * (
          max(0.0, ==b==) + ( ==alpha== * (exp(min(0.0, ==b==)) - 1.0) )
        );
        break;
      case neuron_transfer_function_relu:
        ==a== = fmax(0.0, ==b==);
        break;
    }
  )";
  code = rafko_utilities::replace_all_in_string(code, std::regex("==a=="), a);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==b=="), b);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="), operation_index);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==alpha=="), std::to_string(settings.get_alpha()));
  code = rafko_utilities::replace_all_in_string(code, std::regex("==lambda=="), std::to_string(settings.get_lambda()));
  return code;
}

#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
