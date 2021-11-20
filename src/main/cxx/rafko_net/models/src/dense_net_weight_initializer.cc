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
#include "rafko_net/models/dense_net_weight_initializer.h"

#include <time.h>

#include <cmath>
#include <cstdlib>
#include <algorithm>

#include "rafko_net/models/transfer_function.h"

namespace rafko_net {

using std::min;
using std::max;

using rafko_mainframe::RafkoServiceContext;

sdouble32 DenseNetWeightInitializer::get_weight_amplitude(Transfer_functions used_transfer_function) const{
  sdouble32 amplitude;
  switch(used_transfer_function){
  case transfer_function_elu:
  case transfer_function_relu:
  case transfer_function_selu:
    amplitude = (sqrt(2 / (expected_input_number))); /* Kaiming initialization */
    break;
  default:
    amplitude = (sqrt(2 / (expected_input_number * expected_input_maximum_value)));
    break;
  }
  return max(context.get_epsilon(),amplitude);
}

sdouble32 DenseNetWeightInitializer::next_weight_for(Transfer_functions used_transfer_function) const{
  return ((rand()%2 == 0)?-double_literal(1.0):double_literal(1.0)) * limit_weight(
    (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/get_weight_amplitude(used_transfer_function))))
  );
}

sdouble32 DenseNetWeightInitializer::next_memory_filter() const{
  if(memMin <  memMax){
    sdouble32 diff = memMax - memMin;
    return (double_literal(0.0) == diff)?0:(
       memMin + (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/diff)))
    );
  } else return memMin;
}

sdouble32 DenseNetWeightInitializer::next_bias() const{
  return double_literal(0.0);
}

} /* namespace rafko_net */
