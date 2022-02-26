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

#ifndef RAFKO_SETTINGS_H
#define RAFKO_SETTINGS_H

#include "rafko_global.h"

#include <math.h>
#include <vector>
#include <utility>

#include "rafko_protocol/training.pb.h"

namespace rafko_mainframe{

class RAFKO_FULL_EXPORT RafkoSettings{
public:
  constexpr std::uint16_t get_max_solve_threads() const{
    return max_solve_threads;
  }

  constexpr std::uint16_t get_sqrt_of_solve_threads() const{
    return sqrt_of_solve_threads;
  }

  constexpr std::uint16_t get_max_processing_threads() const{
    return max_processing_threads;
  }

  constexpr std::uint16_t get_sqrt_of_process_threads() const{
    return sqrt_of_process_threads;
  }

  constexpr std::uint32_t get_tolerance_loop_value() const{
    return tolerance_loop_value;
  }

  constexpr double get_device_max_megabytes() const{
    return device_max_megabytes;
  }

  constexpr google::protobuf::Arena* get_arena_ptr() const{
    return arena_ptr;
  }

  double get_learning_rate(std::uint32_t iteration = 0) const;

  constexpr double get_dropout_probability() const{
    return droput_probability;
  }

  std::uint32_t get_minibatch_size() const{
    return hypers.minibatch_size();
  }

  std::uint32_t get_memory_truncation() const{
    return hypers.memory_truncation();
  }

  bool get_training_strategy(rafko_gym::Training_strategy strategy){
    return (0u < (static_cast<std::uint32_t>(hypers.training_strategies()) & static_cast<std::uint32_t>(strategy)));
  }

  double get_alpha() const{
    return hypers.alpha();
  }

  double get_beta() const{
    return hypers.beta();
  }

  double get_beta_2() const{
    return hypers.beta();
  }

  double get_gamma() const{
    return hypers.gamma();
  }

  double get_delta() const{
    return hypers.delta();
  }

  double get_epsilon() const{
    return hypers.epsilon();
  }

  constexpr double get_sqrt_epsilon() const{
    return sqrt_epsilon;
  }

  double get_zetta() const{
    return hypers.zetta();
  }

  double get_lambda() const{
    return hypers.lambda();
  }

  RafkoSettings& set_learning_rate(double learning_rate_){
    hypers.set_learning_rate(learning_rate_);
    calculate_learning_rate_decay();
    return *this;
  }

  RafkoSettings& set_minibatch_size(std::uint32_t minibatch_size_){
    hypers.set_minibatch_size(minibatch_size_);
    return *this;
  }

  RafkoSettings& set_max_solve_threads(double max_solve_threads_){
    max_solve_threads = max_solve_threads_;
    sqrt_of_solve_threads = static_cast<std::uint16_t>(std::max(
      (1.0), std::sqrt(static_cast<double>(max_solve_threads))
    ));
    return *this;
  }

  RafkoSettings& set_max_processing_threads(std::uint16_t max_processing_threads_){
    max_processing_threads = max_processing_threads_;
    sqrt_of_process_threads = static_cast<std::uint16_t>(std::max(
      (1.0), std::sqrt(static_cast<double>(max_processing_threads))
    ));
    return *this;
  }

  constexpr void set_tolerance_loop_value(std::uint32_t tolerance_loop_value_){
    tolerance_loop_value = tolerance_loop_value_;
  }

  constexpr RafkoSettings& set_device_max_megabytes(double device_max_megabytes_){
    device_max_megabytes = device_max_megabytes_;
    return *this;
  }

  constexpr RafkoSettings& set_arena_ptr(google::protobuf::Arena* arena_ptr_){
    arena_ptr = arena_ptr_;
    return *this;
  }

  RafkoSettings& set_memory_truncation(std::uint32_t memory_truncation_){
    hypers.set_memory_truncation(memory_truncation_);
    return *this;
  }

  RafkoSettings& set_alpha(double alpha_){
    hypers.set_epsilon(alpha_);
    return *this;
  }

  RafkoSettings& set_beta(double beta_){
    hypers.set_beta(beta_);
    return *this;
  }

  RafkoSettings& set_beta_2(double beta_){
    hypers.set_beta(beta_);
    return *this;
  }

  RafkoSettings& set_gamma(double gamma_){
    hypers.set_gamma(gamma_);
    return *this;
  }

  RafkoSettings& set_delta(double delta_){
    hypers.set_delta(delta_);
    return *this;
  }

  RafkoSettings& set_epsilon(double epsilon_){
    hypers.set_epsilon(epsilon_);
    sqrt_epsilon = std::sqrt(epsilon_);
    return *this;
  }

  RafkoSettings& set_zetta(double zetta_){
    hypers.set_zetta(zetta_);
    return *this;
  }

  RafkoSettings& set_lambda(double lambda_){
    hypers.set_lambda(lambda_);
    return *this;
  }

  RafkoSettings& set_hypers(rafko_gym::TrainingHyperparameters hypers_){
    hypers.CopyFrom(hypers_);
    return *this;
  }

  RafkoSettings& set_training_strategy(rafko_gym::Training_strategy strategy, bool enable){
    if(enable){
      hypers.set_training_strategies(
        static_cast<rafko_gym::Training_strategy>(static_cast<std::uint32_t>(hypers.training_strategies()) | static_cast<std::uint32_t>(strategy))
      );
    }else{
      hypers.set_training_strategies(
        static_cast<rafko_gym::Training_strategy>(static_cast<std::uint32_t>(hypers.training_strategies()) & (~static_cast<std::uint32_t>(strategy)))
      );
    }
    return *this;
  }

  RafkoSettings& set_learning_rate_decay(std::vector<std::pair<std::uint32_t,double>>&& iteration_with_value){
    learning_rate_decay = std::move(iteration_with_value);
    calculate_learning_rate_decay();
    return *this;
  }

  constexpr RafkoSettings& set_droput_probability(double droput_probability_){
    droput_probability = droput_probability_;
    return *this;
  }

  RafkoSettings(){
    hypers.set_learning_rate((1e-6));
    hypers.set_minibatch_size(64);
    hypers.set_memory_truncation(2);

    hypers.set_alpha((1.6732));
    hypers.set_beta((0.9));
    hypers.set_beta_2((0.99));
    hypers.set_gamma((0.9));
    hypers.set_delta((0.03));
    hypers.set_epsilon(1e-8); /* very small positive value almost greater, than (0.0) */
    hypers.set_zetta((0.3));
    hypers.set_lambda((1.0507));
    hypers.set_training_strategies(rafko_gym::Training_strategy::training_strategy_unknown);
  }

private:
  std::uint16_t max_solve_threads = 4u;
  std::uint16_t sqrt_of_solve_threads = 2u;
  std::uint16_t max_processing_threads = 4u;
  std::uint16_t sqrt_of_process_threads = 2u;
  std::uint32_t tolerance_loop_value = 100u;
  double sqrt_epsilon = std::sqrt((1e-15));
  double device_max_megabytes = (2048);
  google::protobuf::Arena* arena_ptr = nullptr;
  rafko_gym::TrainingHyperparameters hypers = rafko_gym::TrainingHyperparameters();
  mutable std::uint32_t learning_rate_decay_iteration_cache = 0u;
  mutable std::uint32_t learning_rate_decay_index_cache = 0u;
  std::vector<std::pair<std::uint32_t, double>> learning_rate_with_decay;
  std::vector<std::pair<std::uint32_t, double>> learning_rate_decay;
  double droput_probability = (0.2);

  /**
   * @brief      Calculates the learning rates for different iteration indices
   *             based on the decay and the initial learning rate
   */
  void calculate_learning_rate_decay(){
    double learning_rate = get_learning_rate();
    learning_rate_with_decay.clear();
    for(std::pair<std::uint32_t, double> decay : learning_rate_decay){
      learning_rate *= std::get<double>(decay);
      learning_rate_with_decay.push_back({std::get<std::uint32_t>(decay), learning_rate});
    }
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_SETTINGS_H */
