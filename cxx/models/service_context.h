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
private:
  uint16 max_solve_threads = 16;
  uint16 max_processing_threads = 32;
  sdouble32 device_max_megabytes = 2048.0;
  Arena* arena_ptr = nullptr;
};

} /* namespace sparse_net_library */

#endif /* SERVICE_CONTEXT_H */