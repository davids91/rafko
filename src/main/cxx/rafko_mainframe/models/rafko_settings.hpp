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

#include "rafko_global.hpp"

#include <math.h>
#include <vector>
#include <utility>

#include "rafko_protocol/training.pb.h"

namespace rafko_mainframe{

class RAFKO_FULL_EXPORT RafkoSettings{
public:
  constexpr std::uint16_t get_max_solve_threads() const{
    return m_maxSolveThreads;
  }

  constexpr std::uint16_t get_sqrt_of_solve_threads() const{
    return m_sqrtOfSolveThreads;
  }

  constexpr std::uint16_t get_max_processing_threads() const{
    return m_maxProcessingThreads;
  }

  constexpr std::uint16_t get_sqrt_of_process_threads() const{
    return m_sqrtOfProcessThreads;
  }

  constexpr std::uint32_t get_tolerance_loop_value() const{
    return m_toleranceLoopValue;
  }

  constexpr double get_device_max_megabytes() const{
    return m_deviceMaxMegabytes;
  }

  constexpr google::protobuf::Arena* get_arena_ptr() const{
    return m_arenaPtr;
  }

  double get_learning_rate(std::uint32_t iteration = 0) const;

  constexpr double get_dropout_probability() const{
    return m_droputProbability;
  }

  std::uint32_t get_minibatch_size() const{
    return m_hypers.minibatch_size();
  }

  std::uint32_t get_memory_truncation() const{
    return m_hypers.memory_truncation();
  }

  bool get_training_strategy(rafko_gym::Training_strategy strategy) const{
    return (0u < (static_cast<std::uint32_t>(m_hypers.training_strategies()) & static_cast<std::uint32_t>(strategy)));
  }

  double get_alpha() const{
    return m_hypers.alpha();
  }

  double get_beta() const{
    return m_hypers.beta();
  }

  double get_beta_2() const{
    return m_hypers.beta();
  }

  double get_gamma() const{
    return m_hypers.gamma();
  }

  double get_delta() const{
    return m_hypers.delta();
  }

  double get_epsilon() const{
    return m_hypers.epsilon();
  }

  constexpr double get_sqrt_epsilon() const{
    return m_sqrtEpsilon;
  }

  double get_zetta() const{
    return m_hypers.zetta();
  }

  double get_lambda() const{
    return m_hypers.lambda();
  }

  RafkoSettings& set_learning_rate(double learning_rate){
    m_hypers.set_learning_rate(learning_rate);
    calculate_learning_rate_decay();
    return *this;
  }

  RafkoSettings& set_minibatch_size(std::uint32_t minibatch_size){
    m_hypers.set_minibatch_size(minibatch_size);
    return *this;
  }

  RafkoSettings& set_max_solve_threads(double max_solve_threads){
    m_maxSolveThreads = max_solve_threads;
    m_sqrtOfSolveThreads = static_cast<std::uint16_t>(std::max(
      (1.0), std::sqrt(static_cast<double>(m_maxSolveThreads))
    ));
    return *this;
  }

  RafkoSettings& set_max_processing_threads(std::uint16_t max_processing_threads){
    m_maxProcessingThreads = max_processing_threads;
    m_sqrtOfProcessThreads = static_cast<std::uint16_t>(std::max(
      (1.0), std::sqrt(static_cast<double>(m_maxProcessingThreads))
    ));
    return *this;
  }

  constexpr RafkoSettings& set_tolerance_loop_value(std::uint32_t tolerance_loop_value){
    m_toleranceLoopValue = tolerance_loop_value;
    return *this;
  }

  constexpr RafkoSettings& set_device_max_megabytes(double device_max_megabytes){
    m_deviceMaxMegabytes = device_max_megabytes;
    return *this;
  }

  constexpr RafkoSettings& set_arena_ptr(google::protobuf::Arena* arena_ptr){
    m_arenaPtr = arena_ptr;
    return *this;
  }

  RafkoSettings& set_memory_truncation(std::uint32_t memory_truncation){
    m_hypers.set_memory_truncation(memory_truncation);
    return *this;
  }

  RafkoSettings& set_alpha(double alpha){
    m_hypers.set_epsilon(alpha);
    return *this;
  }

  RafkoSettings& set_beta(double beta){
    m_hypers.set_beta(beta);
    return *this;
  }

  RafkoSettings& set_beta_2(double beta){
    m_hypers.set_beta(beta);
    return *this;
  }

  RafkoSettings& set_gamma(double gamma){
    m_hypers.set_gamma(gamma);
    return *this;
  }

  RafkoSettings& set_delta(double delta){
    m_hypers.set_delta(delta);
    return *this;
  }

  RafkoSettings& set_epsilon(double epsilon){
    m_hypers.set_epsilon(epsilon);
    m_sqrtEpsilon = std::sqrt(epsilon);
    return *this;
  }

  RafkoSettings& set_zetta(double zetta){
    m_hypers.set_zetta(zetta);
    return *this;
  }

  RafkoSettings& set_lambda(double lambda){
    m_hypers.set_lambda(lambda);
    return *this;
  }

  RafkoSettings& set_hypers(rafko_gym::TrainingHyperparameters hypers){
    m_hypers.CopyFrom(hypers);
    return *this;
  }

  RafkoSettings& set_training_strategy(rafko_gym::Training_strategy strategy, bool enable){
    if(enable){
      m_hypers.set_training_strategies(
        static_cast<rafko_gym::Training_strategy>(static_cast<std::uint32_t>(m_hypers.training_strategies()) | static_cast<std::uint32_t>(strategy))
      );
    }else{
      m_hypers.set_training_strategies(
        static_cast<rafko_gym::Training_strategy>(static_cast<std::uint32_t>(m_hypers.training_strategies()) & (~static_cast<std::uint32_t>(strategy)))
      );
    }
    return *this;
  }

  RafkoSettings& set_learning_rate_decay(std::vector<std::pair<std::uint32_t,double>>&& iteration_with_value){
    m_learningRateDecay = std::move(iteration_with_value);
    calculate_learning_rate_decay();
    return *this;
  }

  constexpr RafkoSettings& set_droput_probability(double droput_probability){
    m_droputProbability = droput_probability;
    return *this;
  }

  RafkoSettings(){
    m_hypers.set_learning_rate((1e-6));
    m_hypers.set_minibatch_size(64);
    m_hypers.set_memory_truncation(2);

    m_hypers.set_alpha((1.6732));
    m_hypers.set_beta((0.9));
    m_hypers.set_beta_2((0.99));
    m_hypers.set_gamma((0.9));
    m_hypers.set_delta((0.03));
    m_hypers.set_epsilon(1e-8); /* very small positive value almost greater, than (0.0) */
    m_hypers.set_zetta((0.3));
    m_hypers.set_lambda((1.0507));
    m_hypers.set_training_strategies(rafko_gym::Training_strategy::training_strategy_unknown);
  }

private:
  std::uint16_t m_maxSolveThreads = 4u;
  std::uint16_t m_sqrtOfSolveThreads = 2u;
  std::uint16_t m_maxProcessingThreads = 4u;
  std::uint16_t m_sqrtOfProcessThreads = 2u;
  std::uint32_t m_toleranceLoopValue = 100u;
  double m_sqrtEpsilon = std::sqrt((1e-15));
  double m_deviceMaxMegabytes = (2048);
  google::protobuf::Arena* m_arenaPtr = nullptr;
  rafko_gym::TrainingHyperparameters m_hypers = rafko_gym::TrainingHyperparameters();
  mutable std::uint32_t m_learningRateDecayIterationCache = 0u;
  mutable std::uint32_t m_learningRateDecayIndexCache = 0u;
  std::vector<std::pair<std::uint32_t, double>> m_learningRateWithDecay;
  std::vector<std::pair<std::uint32_t, double>> m_learningRateDecay;
  double m_droputProbability = (0.2);

  /**
   * @brief      Calculates the learning rates for different iteration indices
   *             based on the decay and the initial learning rate
   */
  void calculate_learning_rate_decay(){
    double learning_rate = get_learning_rate();
    m_learningRateWithDecay.clear();
    for(std::pair<std::uint32_t, double> decay : m_learningRateDecay){
      learning_rate *= std::get<double>(decay);
      m_learningRateWithDecay.push_back({std::get<std::uint32_t>(decay), learning_rate});
    }
  }
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_SETTINGS_H */
