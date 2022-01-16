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

namespace rafko_net {

Transfer_functions TransferFunction::next(){
  return next({
    transfer_function_identity,
    transfer_function_sigmoid,
    transfer_function_tanh,
    transfer_function_elu,
    transfer_function_selu,
    transfer_function_relu
  });
}

Transfer_functions TransferFunction::next(std::vector<Transfer_functions> range){
  Transfer_functions candidate = static_cast<Transfer_functions>(rand()%Transfer_functions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<Transfer_functions>(rand()%Transfer_functions_ARRAYSIZE);
  return candidate;
}

sdouble32 TransferFunction::get_average_output_range(Transfer_functions function){
  switch(function){
  case transfer_function_sigmoid:
  case transfer_function_tanh:
    return double_literal(1.0);
  case transfer_function_elu:
  case transfer_function_relu:
  case transfer_function_selu:
  case transfer_function_identity:
  default:
    return double_literal(50.0); /* The averagest number there is */
  }
}

sdouble32 TransferFunction::get_value(Transfer_functions function, sdouble32 data) const{
  switch(function){
    case transfer_function_identity: return data; /* Identity means f(x) = x */
    case transfer_function_sigmoid: return double_literal(1.0)/(double_literal(1.0)+exp(-data));
    case transfer_function_tanh: return std::tanh(data);
    case transfer_function_elu:
      if(0 >= data) return settings.get_alpha() * (std::exp(data) -1);
      else return data;
    case transfer_function_selu:
      if(0 >= data) return ((settings.get_alpha() * std::exp(data)) - settings.get_alpha()) * settings.get_lambda();
      else return data;
    case transfer_function_relu: return std::max(double_literal(0.0),data);
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

sdouble32 TransferFunction::get_derivative(Transfer_functions function, sdouble32 data) const{
  switch(function){
    case transfer_function_identity: return double_literal(1.0); /* Identity means f(x) = x */
    case transfer_function_sigmoid: return std::exp(data)/std::pow((std::exp(data) + 1),2);
    case transfer_function_tanh: return std::tanh(data);
    case transfer_function_elu:
      if(0 >= data) return settings.get_alpha() + get_value(function,data);
      else return 1;
    case transfer_function_selu:
      if(0 >= data) return (settings.get_lambda() * settings.get_alpha() * exp(data));
      else return settings.get_lambda();
    case transfer_function_relu:
      if(0 >= data) return 0;
      else return 1;
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

#if(RAFKO_USES_OPENCL)
std::string TransferFunction::get_cl_function_for(Transfer_functions function, std::string x){
  switch(function){
    case transfer_function_identity: return x;
    case transfer_function_sigmoid: return "( 1.0/(1.0 + exp(-1.0 * " + x + ")) )";
    case transfer_function_tanh: return "(tanh(" + x + "))";
    case transfer_function_elu: return "( (max(0.0," + x + ") + min(0.0, " + x + ") * " + std::to_string(settings.get_alpha()) + " )";
    case transfer_function_selu:
    {
      std::string alpha = std::to_string(settings.get_alpha());
      std::string x_negative_component = "min(0.0, " + x + ")";
      std::string x_positive_component = "max(0.0, " + x + ")";
      std::string x_negative_scaled = "((" + alpha + " * exp(" + x_negative_component + ")) - " + alpha + ")";
      return "(" + x_positive_component + " + (" + x_negative_component + " * " + x_negative_scaled + "))";
    }
    case transfer_function_relu: return "max(0.0," + x + ")";
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_net */
