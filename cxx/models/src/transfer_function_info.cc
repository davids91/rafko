#include "models/Transfer_function_info.h"

#include <cmath>

namespace sparse_net_library {

using std::max;

sdouble32 Transfer_function_info::epsilon = 1e-15;
sdouble32 Transfer_function_info::lambda = 1.0507;
sdouble32 Transfer_function_info::alpha = 1.0;

transfer_functions Transfer_function_info::next(){
  return next({
    TRANSFER_FUNCTION_IDENTITY,
    TRANSFER_FUNCTION_SIGMOID,
    TRANSFER_FUNCTION_TANH,
    TRANSFER_FUNCTION_ELU,
    TRANSFER_FUNCTION_SELU,
    TRANSFER_FUNCTION_RELU
  });
}

transfer_functions Transfer_function_info::next(vector<transfer_functions> range){
  transfer_functions candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end()){
    candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  }
  return candidate;
}

sdouble32 Transfer_function_info::get_average_output_range(transfer_functions function){
  switch(function){
  case TRANSFER_FUNCTION_SIGMOID:
  case TRANSFER_FUNCTION_TANH:
    return 1.0;
  case TRANSFER_FUNCTION_ELU:
  case TRANSFER_FUNCTION_RELU:
  case TRANSFER_FUNCTION_SELU:
  case TRANSFER_FUNCTION_IDENTITY:
  default:
    return 50; /* The averagest number there is */
  }
}

void Transfer_function_info::apply_to_data(transfer_functions function, sdouble32& data){
  switch(function){
  case TRANSFER_FUNCTION_IDENTITY: break; /* Identity means f(x) = x */
  case TRANSFER_FUNCTION_SIGMOID:
    data = 1/(1+exp(-data));
    break;
  case TRANSFER_FUNCTION_TANH:
    data = tanh(data);
    break;
  case TRANSFER_FUNCTION_ELU:
    if(0 > data){
      data = alpha * (exp(data) -1);
    }
  case TRANSFER_FUNCTION_SELU:
    data *= lambda;
    break;
  case TRANSFER_FUNCTION_RELU:
    data = max(0.0,data);
    break;
  default: /* unidentified transfer function! */
    throw "Unidentified transfer function queried for information!";
  }
}

} /* namespace sparse_net_library */
