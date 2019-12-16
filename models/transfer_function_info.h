#ifndef Transfer_function_info_H
#define Transfer_function_info_H

#include <vector>

#include "sparsenet_global.h"
#include "models/sNet.pb.h"

namespace sparse_net_library {

class Transfer_function_info
{
public:
  static sdouble32 alpha;
  static sdouble32 lambda;

  static transfer_functions next(std::vector<transfer_functions> range);
  static sdouble32 getAvgOutRange(transfer_functions function);
  static void apply_to_data(transfer_functions function, sdouble32& data);
};

} /* namespace sparse_net_library */
#endif // Transfer_function_info_H
