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

#include <cmath>

namespace rafko_net {

using std::max;

TransferFunctions TransferFunction::next(){
  return next({
    transfer_function_identity,
    transfer_function_sigmoid,
    transfer_function_tanh,
    transfer_function_elu,
    transfer_function_selu,
    transfer_function_relu
  });
}

TransferFunctions TransferFunction::next(vector<TransferFunctions> range){
  TransferFunctions candidate = static_cast<TransferFunctions>(rand()%TransferFunctions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<TransferFunctions>(rand()%TransferFunctions_ARRAYSIZE);
  return candidate;
}

sdouble32 TransferFunction::get_average_output_range(TransferFunctions function){
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

sdouble32 TransferFunction::get_value(TransferFunctions function, sdouble32 data) const{
  switch(function){
    case transfer_function_identity: return data; /* Identity means f(x) = x */
    case transfer_function_sigmoid: return 1/(1+exp(-data));
    case transfer_function_tanh: return tanh(data);
    case transfer_function_elu:
      if(0 >= data) return context.get_alpha() * (exp(data) -1);
      else return data;
    case transfer_function_selu:
      if(0 >= data) return ((context.get_alpha() * exp(data)) - context.get_alpha()) * context.get_lambda();
      else return data;
    case transfer_function_relu: return max(double_literal(0.0),data);
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

sdouble32 TransferFunction::get_derivative(TransferFunctions function, sdouble32 data) const{
  switch(function){
    case transfer_function_identity: return 1; /* Identity means f(x) = x */
    case transfer_function_sigmoid: return exp(data)/pow((exp(data) + 1),2);
    case transfer_function_tanh: return 1/cosh(data);
    case transfer_function_elu:
      if(0 >= data) return context.get_alpha() + get_value(function,data);
      else return 1;
    case transfer_function_selu:
      if(0 >= data) return (context.get_lambda() * context.get_alpha() * exp(data));
      else return context.get_lambda();
    case transfer_function_relu:
      if(0 >= data) return 0;
      else return 1;
    default: throw std::runtime_error("Unidentified transfer function queried for information!");
  }
}

} /* namespace rafko_net */
