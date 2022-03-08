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
    case spike_function_p: return previous_data + ((new_data - previous_data)*parameter);
    case spike_function_amplify_value: return new_data * parameter;
    default: throw std::runtime_error("Unknown spike function requested for calculation!");
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
std::string SpikeFunction::get_cl_function_for(Spike_functions function, std::string new_data, std::string previous_data, std::string parameter){
  switch(function){
    case spike_function_none: return "(" + new_data + ")";
    case spike_function_memory: return "(((" + previous_data + ") * " + parameter + ") + ((" + new_data +") * (1.0 - " + parameter + ")))";
    case spike_function_p: return "(" + previous_data + "+((" + new_data + "-" + previous_data + ") *" + parameter + ")";
    case spike_function_amplify_value: return "(" + new_data + "*" + parameter + ")";
    default: throw std::runtime_error("Unidentified spike function queried for kernel code!");
  }
}

std::string SpikeFunction::get_kernel_function_for(std::string operation_index, std::string previous_data, std::string new_data, std::string parameter){
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
