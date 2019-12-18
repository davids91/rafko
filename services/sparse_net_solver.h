#ifndef sparse_netsolver_H
#define sparse_netsolver_H

#include <memory>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"

namespace sparse_net_library {

using std::vector;
using std::unique_ptr;

class SparseNetSolver
{
public:
  static unique_ptr<sdouble32> solve(SparseNet& net);
  static unique_ptr<sdouble32> solve(vector<sdouble32> output, SparseNet& net);

  static unique_ptr<sdouble32> calculate_spikes(SparseNet& net);
  static unique_ptr<sdouble32> calculate_spikes(vector<sdouble32> output, SparseNet& net);
};

} /* namespace sparse_net_library */
#endif /* sparse_netsolver_H */
