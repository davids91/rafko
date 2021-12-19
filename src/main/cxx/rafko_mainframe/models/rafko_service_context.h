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

#ifndef RAFKO_SERVICE_CONTEXT_H
#define RAFKO_SERVICE_CONTEXT_H

#include "rafko_global.h"

#include <math.h>
#include <vector>
#include <utility>

#include "rafko_protocol/training.pb.h"
#include "rafko_protocol/deep_learning_service.pb.h"

namespace rafko_mainframe{

using std::sqrt;
using google::protobuf::Arena;
using std::vector;
using std::pair;
using std::get;

using rafko_mainframe::ServiceHyperparameters;
using rafko_gym::Training_strategy;

class RAFKO_FULL_EXPORT RafkoServiceContext{
public:
  uint16 get_max_solve_threads() const{
    return max_solve_threads;
  }

  uint16 get_sqrt_of_solve_threads() const{
    return sqrt_of_solve_threads;
  }

  uint16 get_max_processing_threads() const{
    return max_processing_threads;
  }

  uint16 get_sqrt_of_process_threads() const{
    return sqrt_of_process_threads;
  }

  uint32 get_tolerance_loop_value() const{
    return tolerance_loop_value;
  }

  sdouble32 get_device_max_megabytes() const{
    return device_max_megabytes;
  }

  Arena* get_arena_ptr() const{
    return arena_ptr;
  }

  sdouble32 get_learning_rate(uint32 iteration = 0) const{
    if((0 == learning_rate_with_decay.size())||(iteration < get<uint32>(learning_rate_with_decay[0])))
      return hypers.learning_rate();
    if(iteration >= get<uint32>(learning_rate_with_decay.back()))
      return get<sdouble32>(learning_rate_with_decay.back());
    uint32 decay_index = 0;
    if(iteration >= learning_rate_decay_iteration_cache)
      decay_index = learning_rate_decay_index_cache;

    while(
      (decay_index < (learning_rate_with_decay.size()-1u))
      &&(iteration >= get<uint32>(learning_rate_with_decay[decay_index]))
    )++decay_index;

    --decay_index;

    learning_rate_decay_iteration_cache = iteration;
    learning_rate_decay_index_cache = decay_index;

    return get<sdouble32>(learning_rate_with_decay[decay_index]);
  }

  uint32 get_minibatch_size() const{
    return hypers.minibatch_size();
  }

  uint32 get_memory_truncation() const{
    return hypers.memory_truncation();
  }

  bool get_training_strategy(rafko_gym::Training_strategy strategy){
    return (0u < (static_cast<uint32>(hypers.training_strategies()) & static_cast<uint32>(strategy)));
  }

  sdouble32 get_alpha() const{
    return hypers.alpha();
  }

  sdouble32 get_beta() const{
    return hypers.beta();
  }

  sdouble32 get_beta_2() const{
    return hypers.beta();
  }

  sdouble32 get_gamma() const{
    return hypers.gamma();
  }

  sdouble32 get_delta() const{
    return hypers.delta();
  }

  sdouble32 get_epsilon() const{
    return hypers.epsilon();
  }

  sdouble32 get_sqrt_epsilon() const{
    return sqrt_epsilon;
  }

  sdouble32 get_zetta() const{
    return hypers.zetta();
  }

  sdouble32 get_lambda() const{
    return hypers.lambda();
  }

  RafkoServiceContext& set_learning_rate(sdouble32 learning_rate_){
    hypers.set_learning_rate(learning_rate_);
    calculate_learning_rate_decay();
    return *this;
  }

  RafkoServiceContext& set_minibatch_size(uint32 minibatch_size_){
    hypers.set_minibatch_size(minibatch_size_);
    return *this;
  }

  RafkoServiceContext& set_max_solve_threads(sdouble32 max_solve_threads_){
    max_solve_threads = max_solve_threads_;
    sqrt_of_solve_threads = static_cast<uint16>(std::max(
      double_literal(1.0), sqrt(static_cast<sdouble32>(max_solve_threads))
    ));
    return *this;
  }

  RafkoServiceContext& set_max_processing_threads(uint16 max_processing_threads_){
    max_processing_threads = max_processing_threads_;
    sqrt_of_process_threads = static_cast<uint16>(std::max(
      double_literal(1.0), sqrt(static_cast<sdouble32>(max_processing_threads))
    ));
    return *this;
  }

  void set_tolerance_loop_value(uint32 tolerance_loop_value_){
    tolerance_loop_value = tolerance_loop_value_;
  }

  RafkoServiceContext& set_device_max_megabytes(sdouble32 device_max_megabytes_){
    device_max_megabytes = device_max_megabytes_;
    return *this;
  }

  RafkoServiceContext& set_arena_ptr(Arena* arena_ptr_){
    arena_ptr = arena_ptr_;
    return *this;
  }

  RafkoServiceContext& set_memory_truncation(uint32 memory_truncation_){
    hypers.set_memory_truncation(memory_truncation_);
    return *this;
  }

  RafkoServiceContext& set_alpha(sdouble32 alpha_){
    hypers.set_epsilon(alpha_);
    return *this;
  }

  RafkoServiceContext& set_beta(sdouble32 beta_){
    hypers.set_beta(beta_);
    return *this;
  }

  RafkoServiceContext& set_beta_2(sdouble32 beta_){
    hypers.set_beta(beta_);
    return *this;
  }

  RafkoServiceContext& set_gamma(sdouble32 gamma_){
    hypers.set_gamma(gamma_);
    return *this;
  }

  RafkoServiceContext& set_delta(sdouble32 delta_){
    hypers.set_delta(delta_);
    return *this;
  }

  RafkoServiceContext& set_epsilon(sdouble32 epsilon_){
    hypers.set_epsilon(epsilon_);
    sqrt_epsilon = sqrt(epsilon_);
    return *this;
  }

  RafkoServiceContext& set_zetta(sdouble32 zetta_){
    hypers.set_zetta(zetta_);
    return *this;
  }

  RafkoServiceContext& set_lambda(sdouble32 lambda_){
    hypers.set_lambda(lambda_);
    return *this;
  }

  RafkoServiceContext& set_hypers(ServiceHyperparameters hypers_){
    hypers.CopyFrom(hypers_);
    return *this;
  }

  RafkoServiceContext& set_training_strategy(Training_strategy strategy, bool enable){
    if(enable){
      hypers.set_training_strategies(
        static_cast<Training_strategy>(static_cast<uint32>(hypers.training_strategies()) | static_cast<uint32>(strategy))
      );
    }else{
      hypers.set_training_strategies(
        static_cast<Training_strategy>(static_cast<uint32>(hypers.training_strategies()) & (~static_cast<uint32>(strategy)))
      );
    }
    return *this;
  }

  RafkoServiceContext& set_learning_rate_decay(vector<pair<uint32,sdouble32>>&& iteration_with_value){
    learning_rate_decay = std::move(iteration_with_value);
    calculate_learning_rate_decay();
    return *this;
  }

  RafkoServiceContext(){
    hypers.set_learning_rate(double_literal(1e-6));
    hypers.set_minibatch_size(64);
    hypers.set_memory_truncation(2);

    hypers.set_alpha(double_literal(1.6732));
    hypers.set_beta(double_literal(0.9));
    hypers.set_beta_2(double_literal(0.99));
    hypers.set_gamma(double_literal(0.9));
    hypers.set_delta(double_literal(0.03));
    hypers.set_epsilon(1e-8); /* very small positive value almost greater, than double_literal(0.0) */
    hypers.set_zetta(double_literal(0.3));
    hypers.set_lambda(double_literal(1.0507));
    hypers.set_training_strategies(rafko_gym::Training_strategy::training_strategy_unknown);
  }

private:
  uint16 max_solve_threads = 4;
  uint16 sqrt_of_solve_threads = 2;
  uint16 max_processing_threads = 4;
  uint16 sqrt_of_process_threads = 2;
  uint32 tolerance_loop_value = 100;
  sdouble32 sqrt_epsilon = sqrt(double_literal(1e-15));
  sdouble32 device_max_megabytes = double_literal(2048);
  Arena* arena_ptr = nullptr;
  ServiceHyperparameters hypers = ServiceHyperparameters();
  mutable uint32 learning_rate_decay_iteration_cache = 0;
  mutable uint32 learning_rate_decay_index_cache = 0;
  vector<pair<uint32, sdouble32>> learning_rate_with_decay;
  vector<pair<uint32, sdouble32>> learning_rate_decay;


  /**
   * @brief      Calculates the learning rates for different iteration indices
   *             based on the decay and the initial learning rate
   */
  void calculate_learning_rate_decay(){
    sdouble32 learning_rate = get_learning_rate();
    learning_rate_with_decay.clear();
    for(pair<uint32, sdouble32> decay : learning_rate_decay){
      learning_rate *= get<sdouble32>(decay);
      learning_rate_with_decay.push_back({get<uint32>(decay), learning_rate});
    }
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_SERVICE_CONTEXT_H */
