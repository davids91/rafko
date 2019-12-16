#include "models/transferfunctioninfo.h"

namespace sparse_net_library {

transfer_functions TransferFunctionInfo::next(std::vector<transfer_functions> range){
  transfer_functions candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  while(std::find(range.begin(), range.end(), candidate) == range.end()){
    candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  }
  return candidate;
}

sdouble32 TransferFunctionInfo::getAvgOutRange(transfer_functions function){
  switch(function){
  case TRANSFER_FUNC_SIGMOID:
  case TRANSFER_FUNC_TANH:
    return 1.0;
  case TRANSFER_FUNC_RELU:
  case TRANSFER_FUNC_SELU:
  case TRANSFER_FUNC_IDENTITY:
  default:
    return 50; /* The averagest number there is */
  }
}

} /* namespace sparse_net_library */