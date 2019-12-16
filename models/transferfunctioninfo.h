#ifndef TRANSFERFUNCTIONINFO_H
#define TRANSFERFUNCTIONINFO_H

#include <vector>

#include "sparsenet_global.h"
#include "models/sNet.pb.h"

namespace sparse_net_library {

class TransferFunctionInfo
{
public:
  static transfer_functions next(std::vector<transfer_functions> range);
  static sdouble32 getAvgOutRange(transfer_functions function);
};

} /* namespace sparse_net_library */
#endif // TRANSFERFUNCTIONINFO_H
