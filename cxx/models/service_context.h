/*! This file is part of davids91/Rafko.
 *
 *    Rafko is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Rafko is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef SERVICE_CONTEXT_H
#define SERVICE_CONTEXT_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"

namespace sparse_net_library{

using google::protobuf::Arena;

class Service_context{
public:
  uint16 get_max_solve_threads(void) const{
    return max_solve_threads;
  }

  uint16 get_max_processing_threads(void) const{
    return max_processing_threads;
  }

  sdouble32 get_device_max_megabytes(void) const{
    return device_max_megabytes;
  }

  Arena* get_arena_ptr(void) const{
    return arena_ptr;
  }

  sdouble32 get_step_size(void) const{
    return step_size;
  }

  uint32 get_minibatch_size(void) const{
    return minibatch_size;
  }

  uint32 get_memory_truncation(void) const{
    return memory_truncation;
  }

  sdouble32 get_alpha(void) const{
    return alpha;
  }

  sdouble32 get_gamma(void) const{
    return gamma;
  }

  sdouble32 get_epsilon(void) const{
    return epsilon;
  }

  sdouble32 get_lambda(void) const{
    return lambda;
  }

  Service_context& set_step_size(sdouble32 step_size_){
    step_size = step_size_;
    return *this;
  }

  Service_context& set_minibatch_size(uint32 minibatch_size_){
    minibatch_size = minibatch_size_;
    return *this;
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

  Service_context& set_memory_truncation(uint32 memory_truncation_){
    memory_truncation = memory_truncation_;
    return *this;
  }

  Service_context& set_alpha(sdouble32 alpha_){
    epsilon = alpha_;
    return *this;
  }

  Service_context& set_gamma(sdouble32 gamma_){
    gamma = gamma_;
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

private:
  uint16 max_solve_threads = 2;
  uint16 max_processing_threads = 4;
  sdouble32 device_max_megabytes = double_literal(2048);
  Arena* arena_ptr = nullptr;

  sdouble32 step_size = 1e-6;
  uint32 minibatch_size = 64;
  uint32 memory_truncation = 2;

  sdouble32 alpha = double_literal(1.6732);
  sdouble32 gamma = double_literal(0.9);
  sdouble32 epsilon = 1e-15; /* very small positive value almost greater, than double_literal(0.0) */
  sdouble32 lambda = double_literal(1.0507);
};

} /* namespace sparse_net_library */

#endif /* SERVICE_CONTEXT_H */
