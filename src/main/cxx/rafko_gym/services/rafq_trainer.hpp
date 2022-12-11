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
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {}
  )
  : RafQTrainer(
    network, action_count, q_set_size, environment, 
    std::make_shared<RafkoCost>(*m_settings, cost_function), settings
  )
  {
  }

  RafQTrainer(
    rafko_net::RafkoNet& network, std::uint32_t action_count, std::uint32_t q_set_size, 
    std::shared_ptr<RafQEnvironment> environment, std::shared_ptr<rafko_gym::RafkoObjective> objective, 
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {}
  )
  : RafkoAutonomousEntity(settings)
  , m_stableNetwork(network)
  , m_volatileNetwork(google::protobuf::Arena::Create<rafko_net::RafkoNet>(
    m_settings->get_arena_ptr(), m_stableNetwork
  ))
  , m_environment(environment)
  , m_objective(objective)
  , m_qSet(std::make_shared<RafQSet>(
    *m_settings, *m_environment, action_count, q_set_size, m_settings->get_delta()
  ))
  #if(RAFKO_USES_OPENCL)
  , m_context(
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(m_stableNetwork, settings, m_objective)
  )
  , m_optimizer(
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<RafkoAutodiffGPUOptimizer>(settings, *m_volatileNetwork, m_qSet, m_context)
  )
  #else
  , m_context(std::make_shared<rafko_mainframe::RafkoCPUContext>(m_stableNetwork, settings, objective))
  , m_optimizer(std::make_shared<RafkoAutodiffOptimizer>(settings, *m_volatileNetwork, m_context))
  #endif/*(RAFKO_USES_OPENCL)*/
  , m_randomActionGenerator(m_environment->action_properties().m_mean, m_environment->action_properties().m_standardDeviation)
  {
  }

  std::uint32_t q_set_size() const{
    return m_qSet->get_number_of_sequences();
  }

  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u, bool force_gpu_upload = false){
    return m_context->stochastic_evaluation(to_seed, seed_value, force_gpu_upload);
  }

  double full_evaluation(bool force_gpu_upload = false){
    return m_context->full_evaluation(force_gpu_upload);
  }

  void iterate(
    std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs, 
    const std::function<void(double/*progress*/)>& progress_callback = [](double){}
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

  FeatureVector generate_action(const FeatureVector& state, double exploration_ratio);
};

} /* namespace rafko_gym */

#endif /* RAFQ_TRAINER_H */
