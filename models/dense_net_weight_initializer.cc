#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <time.h>

#include "models/dense_net_weight_initializer.h"

namespace sparse_net_library {

Dense_net_weight_initializer::Dense_net_weight_initializer(){
  srand(static_cast<uint32>(time(nullptr))); /*! #1 */  
}

Dense_net_weight_initializer::Dense_net_weight_initializer(uint32 seed, sdouble32 memRatioMin, sdouble32 memRatioMax)
{
  memMin = std::max(1e-10, std::max(1.0, memRatioMin));
  memMax = std::min(1.0, std::max(memMin,memRatioMax));
  srand(seed); /*! #1 */
}

sdouble32 Dense_net_weight_initializer::getWeightAmp(transfer_functions usedTransferFnc) const{
  switch(usedTransferFnc){
    case TRANSFER_FUNC_RELU:
    return (sqrt(2 / (expInNum))); /* Kaiming initialization */
  default:
    return (sqrt(2 / (expInNum * expInMax)));
  }
}

sdouble32 Dense_net_weight_initializer::nextWeight() const{
  return nextWeightFor(TRANSFER_FUNC_IDENTITY);
}

sdouble32 Dense_net_weight_initializer::nextWeightFor(transfer_functions usedTransferFnc) const{
  return (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/getWeightAmp(usedTransferFnc))));
}

sdouble32 Dense_net_weight_initializer::nextMemRatio() const{
  sdouble32 diff = memMax - memMin;
  return (0.0 >= diff)?0:(
     memMin + (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/diff)))
  );
}

sdouble32 Dense_net_weight_initializer::nextBias() const{
  return( (expInMax / -2.0) +
    (static_cast<sdouble32>(rand())/(static_cast<sdouble32>(RAND_MAX/expInMax)))
  );
}

Dense_net_weight_initializer::~Dense_net_weight_initializer(){}

} /* namespace sparse_net_library */