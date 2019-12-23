#ifndef SOLUTION_SOLVER_H
#define SOLUTION_SOLVER_H

#include "sparse_net_global.h"

#include "models/gen/solution.pb.h"

#include <vector>

namespace sparse_net_library{

using std::vector;

class Solution_solver{
public:
  vector<sdouble32> solve(const Solution& solution, vector<sdouble32> input);
};

} /* namespace sparse_net_library */

#endif /* SOLUTION_SOLVER_H */
