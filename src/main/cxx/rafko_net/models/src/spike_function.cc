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

#include "rafko_net/models/spike_function.h"

#include <stdexcept>

#if(RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.h"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_net {

double SpikeFunction::get_value(Spike_functions function, double parameter, double new_data, double previous_data){
  switch(function){
    case spike_function_none: return new_data;
    case spike_function_memory: return (previous_data * parameter) + (new_data * (1.0-parameter));
    case spike_function_p: return previous_data + ((new_data - previous_data) * parameter);
    case spike_function_amplify_value: return new_data * parameter;
    default: throw std::runtime_error("Unknown spike function requested for calculation!");
  }
}

double SpikeFunction::get_derivative_for_w( /* means: x = w; new_data = g(x); previous_data = f(x) */
  Spike_functions function, double parameter,
  double new_data, double new_data_d,
  double previous_data, double previous_data_d
){
  switch(function){
    case spike_function_none: /* S(x,w,f(x),g(x)) = g(x) */
      return new_data_d; /* S'(x,w,f(x),g(x)) = g'(x) */
    case spike_function_memory: /* S(x,w,f(x),g(x)) = w * f(x) + g(x) - w * g(x) */
      /* S'(x,w,f(x),g(x)) =  w * f'(x) + f(x) - w * g'(x)  + g'(x) - g(x) */
      return (parameter * previous_data_d) + previous_data - (parameter * new_data_d) + new_data_d - new_data;
    case spike_function_p: /* S(x,w,f(x),g(x)) = f(x) + g(x) * w - f(x) * w */
      /* S'(x,w,f(x),g(x)) = -w * f'(x) + f'(x) - f(w) + w * g'(x) + g(x) */
      return -parameter * previous_data_d + previous_data_d - previous_data + parameter * new_data_d + new_data;
    case spike_function_amplify_value: /* S(x,w,f(x),g(x)) = w * g(x) */
      /* S'(x,w,f(x),g(x)) = w * g'(x) + g(x) */
      return parameter * new_data_d + new_data;
    default: throw std::runtime_error("Unknown spike function requested for derivative calculation!");
  }
}

double SpikeFunction::get_derivative_not_for_w( /* means: x = w; new_data = g(x); previous_data = f(x) */
  Spike_functions function, double parameter, double new_data_d, double previous_data_d
){
  switch(function){
    case spike_function_none: /* S(x,w,f(x),g(x)) = g(x) */
      return new_data_d;
    case spike_function_memory: /* S(x,w,f(x),g(x)) = w * f(x) + g(x) - w * g(x) */
      /* S'(x,w,f(x),g(x)) = w * f'(x)        - w * g'(x) + g'(x) */
      return (parameter * previous_data_d) - (parameter * new_data_d) + new_data_d;
    case spike_function_p: /* S(x,w,f(x),g(x)) = f(x) + g(x) * w - f(x) * w */
      /* S'(x,w,f(x),g(x)) = w * g'(x) - (w - 1) * f'(x) */
      return (parameter * new_data_d)  - ((parameter - 1.0) * previous_data_d);
    case spike_function_amplify_value: /* S(x,w,f(x),g(x)) = w * g(x) */
      /* S'(x,w,f(x),g(x)) = w * g'(x) */
      return parameter * new_data_d;
    default: throw std::runtime_error("Unknown spike function requested for derivative calculation!");
  }
}

Spike_functions SpikeFunction::next(std::set<Spike_functions> range){
  RFASSERT( 0u < range.size() );
  if(1u == range.size()) return *range.begin();

  Spike_functions candidate = static_cast<Spike_functions>(rand()%Spike_functions_ARRAYSIZE);
  while(!Spike_functions_IsValid(candidate)||find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<Spike_functions>(rand()%Spike_functions_ARRAYSIZE);

  return candidate;
}

#if(RAFKO_USES_OPENCL)
std::string SpikeFunction::get_kernel_function_for(Spike_functions function, std::string new_data, std::string previous_data, std::string parameter){
  switch(function){
    case spike_function_none: return "(" + new_data + ")";
    case spike_function_memory: return "(((" + previous_data + ") * " + parameter + ") + ((" + new_data +") * (1.0 - " + parameter + ")))";
    case spike_function_p: return "(" + previous_data + "+((" + new_data + "-" + previous_data + ") *" + parameter + ")";
    case spike_function_amplify_value: return "(" + new_data + "*" + parameter + ")";
    default: throw std::runtime_error("Unidentified spike function queried for kernel code!");
  }
}

std::string SpikeFunction::get_all_kernel_functions_for(std::string operation_index, std::string previous_data, std::string new_data, std::string parameter){
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
      case neuron_spike_function_amplify_value:
        ==previous_data== = ==parameter== * ==new_data==;
        break;
    }
  )";
  code = rafko_utilities::replace_all_in_string(code, std::regex("==parameter=="), parameter);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==new_data=="), new_data);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==previous_data=="), previous_data);
  code = rafko_utilities::replace_all_in_string(code, std::regex("==op=="), operation_index);
  return code;
}

std::string SpikeFunction::get_derivative_kernel_for_w(
  Spike_functions function, std::string parameter,
  std::string new_data, std::string new_data_d,
  std::string previous_data, std::string previous_data_d
){
  std::string parameter_ = "(" + parameter + ")";
  std::string new_data_d_ = "(" + new_data_d + ")";
  std::string previous_data_d_ = "(" + previous_data_d + ")";
  switch(function){
    case spike_function_none: return new_data_d;
    case spike_function_memory:
      return(
        "(" + parameter_ + "*" + previous_data_d_ + ") + " + previous_data
        + " - ((" + parameter_ + ") * (" + new_data_d_ + ")) + (" + new_data_d_ + ")-(" + new_data + ")"
      );
    case spike_function_p:
      return(
        "-" + parameter_ + " * " + previous_data_d_ + " + " + previous_data_d_ + " - " + previous_data
        + " + " + parameter_ + " * " + new_data_d_ + " + " + new_data
      );
    case spike_function_amplify_value:
      return "(" + parameter_ + "*" + new_data_d_ + "+" + new_data + ")";
    default: throw std::runtime_error("Unknown spike function requested for derivative calculation!");
  }
}

std::string SpikeFunction::get_derivative_kernel_not_for_w(
  Spike_functions function, std::string parameter,
  std::string new_data_d, std::string previous_data_d
){
  std::string parameter_ = "(" + parameter + ")";
  std::string new_data_d_ = "(" + new_data_d + ")";
  std::string previous_data_d_ = "(" + previous_data_d + ")";
  switch(function){
    case spike_function_none: return new_data_d_;
    case spike_function_memory:
    return(
      "(" + parameter_ + "*" + previous_data_d_ + ")" + " - (" + parameter_ + "*" + new_data_d_ + ")" + "+" + new_data_d_
    );
    case spike_function_p:
    return "(" + parameter_ + "*" + new_data_d_ + ")" + " - ((" + parameter_ + "-1.0) * " + previous_data_d_ + ")";
    case spike_function_amplify_value:
      return "(" + parameter_ + "*" + new_data_d_ + ")";
    default: throw std::runtime_error("Unknown spike function requested for derivative calculation!");
  }
}

std::string SpikeFunction::get_kernel_enum_for(Spike_functions function){
  switch(function){
    case spike_function_none: return "neuron_spike_function_none";
    case spike_function_memory: return "neuron_spike_function_memory";
    case spike_function_p: return "neuron_spike_function_p";
    case spike_function_amplify_value: return "neuron_spike_function_amplify_value";
    default: throw std::runtime_error("Unidentified spike function queried for information!");
  }
}

#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
