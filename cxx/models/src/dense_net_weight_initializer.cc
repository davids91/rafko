#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <time.h>

#include "models/dense_net_weight_initializer.h"
#include "models/transfer_function.h"

namespace sparse_net_library {

using std::min;
using std::max;

Dense_net_weight_initializer::Dense_net_weight_initializer(bool seed){
  if(seed)srand(static_cast<uint32>(time(nullptr)));
}

Dense_net_weight_initializer::Dense_net_weight_initializer(sdouble32 memRatioMin, sdouble32 memRatioMax){
  memMin = max(context.get_epsilon(), min(1.0, memRatioMin));
  memMax = min(1.0, max(memMin,memRatioMax));
}

Dense_net_weight_initializer::Dense_net_weight_initializer(uint32 seed, sdouble32 memRatioMin, sdouble32 memRatioMax){
  Dense_net_weight_initializer(memRatioMin,memRatioMax);
  srand(seed);
}

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
  return ((rand()%2 == 0)?-1.0:1.0) * limit_weight( 
    (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/get_weight_amplitude(used_transfer_function))))
  );
}

sdouble32 Dense_net_weight_initializer::next_memory_filter() const{
  if(memMin <  memMax){
    sdouble32 diff = memMax - memMin;
    return (0.0 == diff)?0:(
       memMin + (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/diff)))
    );
  } else return memMin;
}

sdouble32 Dense_net_weight_initializer::next_bias() const{
  return ( (expected_input_maximum_value / -2.0) +
    (static_cast<sdouble32>(rand()%static_cast<int>(expected_input_maximum_value)))
  );
}

} /* namespace sparse_net_library */