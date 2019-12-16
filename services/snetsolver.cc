#include "services/snetsolver.h"

namespace sparse_net_library {

std::unique_ptr<sdouble32> snetsolver::solve(SparseNet const * net){
  return solve(std::unique_ptr<sdouble32[]>(new sdouble32[net->output_neuron_number()]),net);
}

std::unique_ptr<sdouble32> snetsolver::solve(std::unique_ptr<sdouble32[]> output, const SparseNet *net){

  throw NOT_IMPLEMENTED_EXCEPTION;
}

std::unique_ptr<sdouble32> snetsolver::calculateSpikes(SparseNet const * net){
  return calculateSpikes(std::unique_ptr<sdouble32[]>(new sdouble32[net->output_neuron_number()]),net);
}

std::unique_ptr<sdouble32> snetsolver::calculateSpikes(std::unique_ptr<sdouble32[]> output, const SparseNet *net){

  throw NOT_IMPLEMENTED_EXCEPTION;
}

} /* namespace sparse_net_library */