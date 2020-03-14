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

#include "models/transfer_function.h"

#include <cmath>

namespace sparse_net_library {

using std::max;

transfer_functions Transfer_function::next(){
  return next({
    TRANSFER_FUNCTION_IDENTITY,
    TRANSFER_FUNCTION_SIGMOID,
    TRANSFER_FUNCTION_TANH,
    TRANSFER_FUNCTION_ELU,
    TRANSFER_FUNCTION_SELU,
    TRANSFER_FUNCTION_RELU
  });
}

transfer_functions Transfer_function::next(vector<transfer_functions> range){
  transfer_functions candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  return candidate;
}

sdouble32 Transfer_function::get_average_output_range(transfer_functions function){
  switch(function){
  case TRANSFER_FUNCTION_SIGMOID:
  case TRANSFER_FUNCTION_TANH:
    return 1.0L;
  case TRANSFER_FUNCTION_ELU:
  case TRANSFER_FUNCTION_RELU:
  case TRANSFER_FUNCTION_SELU:
  case TRANSFER_FUNCTION_IDENTITY:
  default:
    return 50.0L; /* The averagest number there is */
  }
}

sdouble32 Transfer_function::get_value(transfer_functions function, sdouble32 data){
  switch(function){
    case TRANSFER_FUNCTION_IDENTITY: return data; /* Identity means f(x) = x */
    case TRANSFER_FUNCTION_SIGMOID: return 1/(1+exp(-data));
    case TRANSFER_FUNCTION_TANH: return tanh(data);
    case TRANSFER_FUNCTION_ELU:
      if(0 > data) return context.get_alpha() * (exp(data) -1);
      else return data;
    case TRANSFER_FUNCTION_SELU:
      if(0 > data) return context.get_alpha() * (exp(data) -1) * context.get_lambda();
      else return data;
    case TRANSFER_FUNCTION_RELU: return max(0.0L,data);
    default: throw "Unidentified transfer function queried for information!";
  }
}

sdouble32 Transfer_function::get_derivative(transfer_functions function, sdouble32 data){
  switch(function){
    case TRANSFER_FUNCTION_IDENTITY: return 1; /* Identity means f(x) = x */
    case TRANSFER_FUNCTION_SIGMOID: return exp(data)/pow((exp(data) + 1),2);
    case TRANSFER_FUNCTION_TANH: return 1/cosh(data);
    case TRANSFER_FUNCTION_ELU:
    case TRANSFER_FUNCTION_SELU:
      if(0 >= data) return context.get_alpha() * exp(data);
      else return 1;
    case TRANSFER_FUNCTION_RELU:
      if(0 >= data) return 0;
      else return 1;
    default: throw "Unidentified transfer function queried for information!";
  }
}

} /* namespace sparse_net_library */
