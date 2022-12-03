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
#include <algorithm>
#include <optional>
#include <functional>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_mainframe/models/rafko_autonomous_entity.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/services/solution_solver.hpp"

#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"
#include "rafko_mainframe/services/rafko_gpu_context.hpp"
#include "rafko_gym/services/rafko_autodiff_gpu_optimizer.hpp"
#else
#include "rafko_mainframe/services/rafko_cpu_context.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"
#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafq_set.hpp"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

namespace rafko_gym{
//TODO: Move implementations to different files, maybe rid of the template parameter
/**
 * @brief      Trainer facilitating QLearning 
 */
class RAFKO_EXPORT RafQTrainer : public rafko_mainframe::RafkoAutonomousEntity{
  using MaybeDataType = RafQEnvironment::MaybeDataType; 
public:
  using DataType = RafQEnvironment::DataType; //TODO: Rename to FeatureVecType? or something like that
  using DataView = RafQEnvironment::DataView;

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
    const std::function<void(double/*progress*/)>& progress_callback = {}
  ){
    std::vector<DataType> xp_states;
    std::vector<DataType> xp_actions;
    std::uint32_t discovery_iteration = 0;
    if(0 < max_discovery_length){
      bool terminal = false;
      m_environment->reset();
      RFASSERT(m_environment->current_state().has_value());
      xp_states.push_back(m_environment->current_state().value().get());
      RFASSERT(xp_states.back().size() == m_environment->state_size());
      xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
      RFASSERT(xp_actions.back().size() == m_environment->action_size());
      [[maybe_unused]] std::uint32_t xp_states_size = 0;
      [[maybe_unused]] std::uint32_t xp_actions_size = 0;
      while(!terminal && (discovery_iteration < max_discovery_length)){
        RFASSERT(xp_states_size < xp_states.size());
        RFASSERT(xp_actions_size < xp_actions.size());
        xp_states_size = xp_states.size();
        xp_actions_size = xp_actions.size();

        auto next_state = m_environment->next(xp_actions.back());
        terminal = next_state.m_terminal && next_state.m_resultState.has_value();
        xp_actions.back() = RafQSetItemConstView::action_slot(xp_actions.back(), next_state.m_resultQValue);
        RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
        if(!terminal && next_state.m_resultState.has_value() && (discovery_iteration < (max_discovery_length - 1))){
          xp_states.push_back(next_state.m_resultState.value()); //TODO: bad alloc?!?!
          RFASSERT(xp_states.back().size() == m_environment->state_size());
          xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
          RFASSERT(xp_actions.back().size() == m_environment->action_size());
        }else break;
        progress_callback(static_cast<double>(discovery_iteration) / static_cast<double>(max_discovery_length + q_set_training_epochs));
        ++discovery_iteration;
      }
      RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
    }
  
    std::uint32_t initial_q_set_size = m_qSet->get_number_of_sequences();  
    if((0 < xp_states.size()) && (0 < xp_actions.size())){
      m_qSet->incorporate(xp_states, xp_actions);
      if(0 < q_set_training_epochs)
        m_optimizer->build(m_qSet, m_objective);
    }
    if((0 < q_set_training_epochs) && (0 == initial_q_set_size)){
      m_optimizer->build(m_qSet, m_objective);
    }
    for(std::uint32_t training_iteration = 0; training_iteration < q_set_training_epochs; ++training_iteration){
      m_optimizer->iterate(*m_qSet, (0 == training_iteration)/*force_gpu_upload*/);
      progress_callback(static_cast<double>(discovery_iteration + training_iteration) / static_cast<double>(max_discovery_length + q_set_training_epochs));
    }
    if(0 == (m_iteration % m_settings->get_training_relevant_loop_count())){
      //TODO: Handle modified structure as well
      m_stableNetwork.mutable_weight_table()->Assign(m_volatileNetwork->weight_table().begin(), m_volatileNetwork->weight_table().end());
      m_context->refresh_solution_weights();
    }
    ++m_iteration;
  }

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

  DataType generate_action(const DataType& state, double exploration_ratio){
    static std::random_device rd{};
    static std::mt19937 gen{rd()};
    auto policy_actions = m_context->solve(state, true/*reset_neuron_data*/); /* The context is using the stable network */
    DataType action_for_state = {policy_actions.begin(), policy_actions.begin() + m_environment->action_size()};

    // std::cout << "generated action: ";
    // for(double d : action_for_state) std::cout << "[" << d << "]";

    if((100 * exploration_ratio) > (rand()%100)){ /* Explore! Add Random actions */
      std::uint32_t action_item_index = 0u;
      std::uint32_t random_action_count = 0u;
      std::generate_n(
        action_for_state.begin(), m_environment->action_size(), 
        [this, &action_for_state, &action_item_index, &random_action_count, exploration_ratio]()->double{ 
          ++action_item_index;
          if((100 * exploration_ratio) > (rand()%100)){
            ++random_action_count;
            return m_randomActionGenerator(gen); 
          }
          return action_for_state[action_item_index - 1];
        }
      );
      if(0 == random_action_count)
        action_for_state[rand()%action_for_state.size()] = m_randomActionGenerator(gen);
      // std::cout << "action size: " << action_for_state.size() << std::endl;
    } /* else --> Exploit! Get best action for current state */
    // std::cout << "-->";
    // for(double d : action_for_state) std::cout << "[" << d << "]";
    // std::cout << std::endl;
    return action_for_state;
  }
};

} /* namespace rafko_gym */

#endif /* RAFQ_TRAINER_H */
