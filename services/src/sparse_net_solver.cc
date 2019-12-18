#include "services/sparse_net_solver.h"

namespace sparse_net_library {

unique_ptr<sdouble32> SparseNetSolver::solve(SparseNet& net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

unique_ptr<sdouble32> SparseNetSolver::solve(vector<sdouble32> output, SparseNet& net){

  throw NOT_IMPLEMENTED_EXCEPTION;
}

unique_ptr<sdouble32> SparseNetSolver::calculate_spikes(SparseNet& net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

unique_ptr<sdouble32> SparseNetSolver::calculate_spikes(vector<sdouble32> output, SparseNet& net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

} /* namespace sparse_net_library */
