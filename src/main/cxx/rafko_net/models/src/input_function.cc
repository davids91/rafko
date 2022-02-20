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

namespace rafko_net {

Input_functions InputFunction::next(std::set<Input_functions> range){
  assert( 0u < range.size() );
  if(1u == range.size()) return *range.begin();

  Input_functions candidate = static_cast<Input_functions>(rand()%Input_functions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<Input_functions>(rand()%Input_functions_ARRAYSIZE);

  return candidate;
}

sdouble32 InputFunction::collect(Input_functions function, sdouble32 a, sdouble32 b){
  switch(function){
    case input_function_add: return a + b;
    case input_function_multiply: return a * b;
    default: throw std::runtime_error("Unidentified Input function called!");
  };
}

std::string InputFunction::get_kernel_function_for(Input_functions function, std::string element){
  switch(function){
    case input_function_add: return "+ (" + element + ")";
    case input_function_multiply: return  "* (" + element + ")";
    default: throw std::runtime_error("Unidentified Input function called!");
  };
}


} /* namespace rafko_net */
