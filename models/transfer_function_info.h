#ifndef Transfer_function_info_H
#define Transfer_function_info_H

#include <vector>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"

namespace sparse_net_library {

using std::vector;

class Transfer_function_info
{
public:
  static sdouble32 alpha;
  static sdouble32 lambda;

  static transfer_functions next(vector<transfer_functions> range);
  static sdouble32 getAvgOutRange(transfer_functions function);
  static void apply_to_data(transfer_functions function, sdouble32& data);
};

} /* namespace sparse_net_library */
#endif // Transfer_function_info_H
