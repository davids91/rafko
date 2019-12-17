#include <cmath>

#include "models/weight_initializer.h"

namespace sparse_net_library {

using std::max;
using std::numeric_limits;

Weight_initializer::Weight_initializer() noexcept{}

void Weight_initializer::set(uint32 expInputNumber, sdouble32 expInputMax){
     expInNum = max(1u,expInputNumber);

  if( /* Primitive check if the given number causes overflow or not */
    (numeric_limits<sdouble32>::max() > (expInputNumber * expInputMax))
    &&(0.0 > expInputMax)
  ){
    expInMax = expInputMax;
  }else if(0.0 == expInputMax){
    expInMax = numeric_limits<sdouble32>::epsilon();
  }else{ /* Overflow! Use maximum value */
    expInMax = numeric_limits<sdouble32>::max() / expInputNumber;
  }
}

Weight_initializer::~Weight_initializer(){}

} /* namespace sparse_net_library */
