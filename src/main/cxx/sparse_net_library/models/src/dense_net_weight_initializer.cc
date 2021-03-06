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
#include "sparse_net_library/models/dense_net_weight_initializer.h"

#include <time.h>

#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "sparse_net_library/models/transfer_function.h"

namespace sparse_net_library {

using std::min;
using std::max;

using rafko_mainframe::Service_context;

sdouble32 Dense_net_weight_initializer::get_weight_amplitude(transfer_functions used_transfer_function) const{
  sdouble32 amplitude;
  switch(used_transfer_function){
    case TRANSFER_FUNCTION_RELU:
    amplitude = (sqrt(2 / (expected_input_number))); /* Kaiming initialization */
  default:
    amplitude = (sqrt(2 / (expected_input_number * expected_input_maximum_value)));
  }
  return max(context.get_epsilon(),amplitude);
}

sdouble32 Dense_net_weight_initializer::next_weight_for(transfer_functions used_transfer_function) const{
  return ((rand()%2 == 0)?-double_literal(1.0):double_literal(1.0)) * limit_weight(
    (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/get_weight_amplitude(used_transfer_function))))
  );
}

sdouble32 Dense_net_weight_initializer::next_memory_filter() const{
  if(memMin <  memMax){
    sdouble32 diff = memMax - memMin;
    return (double_literal(0.0) == diff)?0:(
       memMin + (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/diff)))
    );
  } else return memMin;
}

sdouble32 Dense_net_weight_initializer::next_bias() const{
  return double_literal(0.0);
}

} /* namespace sparse_net_library */
