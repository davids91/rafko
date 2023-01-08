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

#ifndef RAFQ_TRAINER_H
#define RAFQ_TRAINER_H

#include "rafko_global.hpp"

#include <memory>
#include <vector>
#include <random>
#include <optional>
#include <functional>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_mainframe/models/rafko_autonomous_entity.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"

#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"
#include "rafko_mainframe/services/rafko_gpu_context.hpp"
#include "rafko_gym/services/rafko_autodiff_gpu_optimizer.hpp"
#else
#include "rafko_mainframe/services/rafko_cpu_context.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafq_set.hpp"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

namespace rafko_gym{
/**
 * @brief      Trainer facilitating QLearning 
 */
class RAFKO_EXPORT RafQTrainer : public rafko_mainframe::RafkoAutonomousEntity{
  using MaybeFeatureVector = RafQEnvironment::MaybeFeatureVector; 
public:
  using FeatureVector = RafQEnvironment::FeatureVector;
  using FeatureView = RafQEnvironment::FeatureView;

  RafQTrainer(
    rafko_net::RafkoNet& network, std::uint32_t action_count, std::uint32_t q_set_size, 
    std::shared_ptr<RafQEnvironment> environment, Cost_functions cost_function,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings
  );

  RafQTrainer(
    rafko_net::RafkoNet& network, std::uint32_t action_count, std::uint32_t q_set_size, 
    std::shared_ptr<RafQEnvironment> environment, std::shared_ptr<rafko_gym::RafkoObjective> objective,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings
  );

  RafQTrainer(
    rafko_net::RafkoNet& network, std::shared_ptr<RafQSet> q_set, 
    std::shared_ptr<RafQEnvironment> environment, Cost_functions cost_function,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings
  );

  RafQTrainer(
    rafko_net::RafkoNet& network, std::shared_ptr<RafQSet> q_set, 
    std::shared_ptr<RafQEnvironment> environment, std::shared_ptr<rafko_gym::RafkoObjective> objective, 
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {}
  );

  /**
   * @brief   provides the current size of the enclosing q-set
   * 
   * @return    number of state-actions pairs stored in the q-set
   */
  std::uint32_t q_set_size() const{
    return m_qSet->get_number_of_sequences();
  }

  /**
   * @brief     Provides const access to the contained Q-set
   * 
   * @return    const reference to the contained Q-set
   */
  const RafQSet& q_set();

  /**
   * @brief     Sets the type of the weight updater used with the enclosed optimizer
   * 
   * @param     updater     The type of the weight updater to set
   */
  void set_weight_updater(rafko_gym::Weight_updaters updater);

  /**
   * @brief   evaluates the stored network on the enclosing q-set
   * 
   * @param[in]      to_seed              A helper value to make Stochastic evaluation deterministicly reproducible
   * @param[in]      seed_value           A helper value to make Stochastic evaluation deterministicly reproducible
   * @param[in]      force_gpu_upload     If set true, data in stored objects are uploaded to GPU regardless of previous uploads
   *                                      Applies only to implementations targeting GPUs
   * 
   * @return    error/fitness value resulting from the evaluation
   */
  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u, bool force_gpu_upload = false){
    if(force_gpu_upload)
      m_context->refresh_solution_weights();
    return m_context->stochastic_evaluation(to_seed, seed_value, force_gpu_upload);
  }

  /**
   * @brief   evaluates the stored network on the enclosing q-set
   * 
   * @param[in]      force_gpu_upload     If set true, data in stored objects are uploaded to GPU regardless of previous uploads
   *                                      Applies only to implementations targeting GPUs
   * 
   * @return    error/fitness value resulting from the evaluation
   */
  double full_evaluation(bool force_gpu_upload = false){
    if(force_gpu_upload)
      m_context->refresh_solution_weights();
    return m_context->full_evaluation(force_gpu_upload);
  }

  /**
   * @brief   Applies one iteration of collecting experience data, incorporating it into the q-set and optimizing the enclosed network for it
   *          It does not reset the environment, so it may be set to any desired initial state before calling the function.
   * 
   * @param[in]     max_discovery_length      Number of discovery steps to take in this iteration
   * @param[in]     exploration_ratio         Exploration vs Exploitation ratio: 1.0 to explore, 0.0 to exploit fully
   * @param[in]     q_set_training_epochs     Number of training iterations to run on the enclosing policy network
   * @param[in]     progress_callback         A function to help with showing progress, providing the actual progress, and step index
   * 
   * @return    error/fitness value resulting from the evaluation
   */
  void iterate(
    std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs, 
    const std::function<void(double/*progress*/, std::uint32_t/*step*/)>& progress_callback = [](double, std::uint32_t){}
  );

private:
  rafko_net::RafkoNet& m_stableNetwork;
  rafko_net::RafkoNet* m_volatileNetwork;
  std::shared_ptr<RafQEnvironment> m_environment;
  std::shared_ptr<RafkoObjective> m_objective;
  std::shared_ptr<RafQSet> m_qSet;
  std::shared_ptr<rafko_mainframe::RafkoContext> m_context;
  std::shared_ptr<RafkoAutodiffOptimizer> m_optimizer;
  std::normal_distribution<double> m_randomActionGenerator;
  std::uint32_t m_iteration = 0;

  /**
   * @brief     Generates an action for the given state and exploration ratio
   * 
   * @param[in]     state                 The state to generate the action to
   * @param[in]     exploration_ratio     Exploration vs Exploitation ratio: 1.0 to explore, 0.0 to exploit fully
   */
  FeatureVector generate_action(const FeatureVector& state, double exploration_ratio);
};

} /* namespace rafko_gym */

#endif /* RAFQ_TRAINER_H */
