#include "models/transfer_function.h"

#include <cmath>

namespace sparse_net_library {

using std::max;

sdouble32 epsilon = 1e-15;
sdouble32 lambda = 1.0507;
sdouble32 alpha = 1.6732;

transfer_functions Transfer_function::next(){
  return next({
    TRANSFER_FUNCTION_IDENTITY,
    TRANSFER_FUNCTION_SIGMOID,
    TRANSFER_FUNCTION_TANH,
    TRANSFER_FUNCTION_ELU,
    TRANSFER_FUNCTION_SELU,
    TRANSFER_FUNCTION_RELU
  });
}

transfer_functions Transfer_function::next(vector<transfer_functions> range){
  transfer_functions candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  while(find(range.begin(), range.end(), candidate) == range.end())
    candidate = static_cast<transfer_functions>(rand()%transfer_functions_ARRAYSIZE);
  return candidate;
}

sdouble32 Transfer_function::get_average_output_range(transfer_functions function){
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

void Transfer_function::apply_to_data(transfer_functions function, sdouble32& data){
  switch(function){
    case TRANSFER_FUNCTION_IDENTITY: break; /* Identity means f(x) = x */
    case TRANSFER_FUNCTION_SIGMOID: data = 1/(1+exp(-data)); break;
    case TRANSFER_FUNCTION_TANH: data = tanh(data); break;
    case TRANSFER_FUNCTION_ELU: if(0 > data) data = alpha * (exp(data) -1);
    case TRANSFER_FUNCTION_SELU: data *= lambda; break;
    case TRANSFER_FUNCTION_RELU: data = max(0.0,data); break;
    default: throw "Unidentified transfer function queried for information!";
  }
}

sdouble32 Transfer_function::apply_derivative(transfer_functions function, sdouble32& data){
  switch(function){
    case TRANSFER_FUNCTION_IDENTITY: return 1; /* Identity means f(x) = x */
    case TRANSFER_FUNCTION_SIGMOID: return exp(data)/pow((exp(data) + 1),2);
    case TRANSFER_FUNCTION_TANH: return 1/cosh(data);
    case TRANSFER_FUNCTION_ELU:
    case TRANSFER_FUNCTION_SELU:
      if(0 > data) return alpha * exp(data);
      else return 1;
    case TRANSFER_FUNCTION_RELU:
      if(0 > data) return 0;
      else return 1;
    default: throw "Unidentified transfer function queried for information!";
  }
}

} /* namespace sparse_net_library */
