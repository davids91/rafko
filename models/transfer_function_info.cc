#include "models/Transfer_function_info.h"

#include <cmath>

namespace sparse_net_library {

sdouble32 Transfer_function_info::lambda = 1.0507;
sdouble32 Transfer_function_info::alpha = 1.0;

transfer_functions Transfer_function_info::next(std::vector<transfer_functions> range){
  transfer_functions candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  while(std::find(range.begin(), range.end(), candidate) == range.end()){
    candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  }
  return candidate;
}

sdouble32 Transfer_function_info::getAvgOutRange(transfer_functions function){
  switch(function){
  case TRANSFER_FUNC_SIGMOID:
  case TRANSFER_FUNC_TANH:
    return 1.0;
  case TRANSFER_FUNC_ELU:
  case TRANSFER_FUNC_RELU:
  case TRANSFER_FUNC_SELU:
  case TRANSFER_FUNC_IDENTITY:
  default:
    return 50; /* The averagest number there is */
  }
}

void Transfer_function_info::apply_to_data(transfer_functions function, sdouble32& data){
  switch(function){
  case TRANSFER_FUNC_SIGMOID:
    data = 1/(1+exp(-data));
    break;
  case TRANSFER_FUNC_TANH:
    data = tanh(data);
    break;
  case TRANSFER_FUNC_RELU:
    data = std::max(0.0,data);
    break;
  case TRANSFER_FUNC_ELU:
    if(0 > data){
      data = alpha * (exp(data) -1);
    }
  case TRANSFER_FUNC_SELU:
    data *= lambda;
    break;
  case TRANSFER_FUNC_IDENTITY: break; /* Identity means f(x) = x */
  default: /* unidentified transfer function! */
    throw UNIDENTIFIED_OPERATION_EXCEPTION;
  }
}

} /* namespace sparse_net_library */
