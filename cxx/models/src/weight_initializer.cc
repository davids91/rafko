#include <cmath>

#include "models/weight_initializer.h"

namespace sparse_net_library {

using std::min;
using std::max;
using std::numeric_limits;

Weight_initializer::Weight_initializer() noexcept{}

void Weight_initializer::set(uint32 expected_input_number_, sdouble32 expected_input_maximum_value_){
  expected_input_number = max(1u,expected_input_number_);

  if( /* Primitive check if the given number causes overflow or not */
    (numeric_limits<sdouble32>::max() > (expected_input_number_ * abs(expected_input_maximum_value_)))
  ){
    expected_input_maximum_value = expected_input_maximum_value_;
  }else if(0.0 == expected_input_maximum_value_){
    expected_input_maximum_value = numeric_limits<sdouble32>::epsilon();
  }else{ /* Overflow! Use maximum value */
    expected_input_maximum_value = numeric_limits<sdouble32>::max() / expected_input_number_;
  }
}

sdouble32 Weight_initializer::next_weight() const{
  return next_weight_for(TRANSFER_FUNCTION_IDENTITY);
}

sdouble32 Weight_initializer::limit_weight(sdouble32 weight) const {
  return min(1.0,max(-1.0,weight));
}

} /* namespace sparse_net_library */
