#ifndef SERVICE_CONTEXT_H
#define SERVICE_CONTEXT_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"

namespace sparse_net_library{

using google::protobuf::Arena;

class Service_context{
public:
  uint16 get_max_solve_threads() const{
    return max_solve_threads;
  }

  uint16 get_max_processing_threads() const{
    return max_processing_threads;
  }

  sdouble32 get_device_max_megabytes() const{
    return device_max_megabytes;
  }

  Arena* get_arena_ptr() const{
    return arena_ptr;
  }

  sdouble32 get_epsilon() const{
    return epsilon;
  }

  sdouble32 get_lambda() const{
    return lambda;
  }

  sdouble32 get_alpha() const{
    return alpha;
  }

  Service_context& set_max_solve_threads(sdouble32 max_solve_threads_){
    max_solve_threads = max_solve_threads_;
    return *this;
  }

  Service_context& set_max_processing_threads(uint16 max_processing_threads_){
    max_processing_threads = max_processing_threads_;
    return *this;
  }

  Service_context& set_device_max_megabytes(sdouble32 device_max_megabytes_){
    device_max_megabytes = device_max_megabytes_;
    return *this;
  }

  Service_context& set_arena_ptr(Arena* arena_ptr_){
    arena_ptr = arena_ptr_;
    return *this;
  }

  Service_context& set_epsilon(sdouble32 epsilon_){
    epsilon = epsilon_;
    return *this;
  }

  Service_context& set_lambda(sdouble32 lambda_){
    epsilon = lambda_;
    return *this;
  }

  Service_context& set_alpha(sdouble32 alpha_){
    epsilon = alpha_;
    return *this;
  }

private:
  uint16 max_solve_threads = 16;
  uint16 max_processing_threads = 32;
  sdouble32 device_max_megabytes = 2048.0;
  Arena* arena_ptr = nullptr;

  sdouble32 epsilon = 1e-15; /* very small positive value almost greater, than 0.0 */
  sdouble32 lambda = 1.0507;
  sdouble32 alpha = 1.6732;
};

} /* namespace sparse_net_library */

#endif /* SERVICE_CONTEXT_H */