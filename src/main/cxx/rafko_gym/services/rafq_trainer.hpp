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

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_mainframe/models/rafko_autonomous_entity.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/services/solution_solver.hpp"

#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafq_set.hpp"
#include "rafko_gym/models/rafko_cost.hpp"
#include "rafko_gym/services/rafko_autodiff_optimizer.hpp"

namespace rafko_gym{
//TODO: Move implementations to different files, maybe rid of the template parameter
/**
 * @brief      Trainer facilitating QLearning 
 */
template<std::size_t ActionCount>
class RAFKO_EXPORT RafQTrainer : public rafko_mainframe::RafkoAutonomousEntity{
  using XPSlot = RafQSetItemConstView<1>;
  using MaybeDataType = RafQEnvironment::MaybeDataType; 
public:
  using DataType = RafQEnvironment::DataType;
  using DataView = RafQEnvironment::DataView;

  RafQTrainer(
    rafko_net::RafkoNet& network, std::uint32_t q_set_size, 
    std::shared_ptr<RafQEnvironment> environment, 
    std::shared_ptr<rafko_gym::RafkoAutodiffOptimizer> optimizer, 
    Cost_functions cost_function, std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {}
  )
  : RafkoAutonomousEntity(settings)
  , m_network(network)
  , m_solverFactory(m_network, m_settings)
  , m_agent(m_solverFactory.build())
  , m_environment(environment)
  , m_objective(*m_settings, cost_function)
  , m_qSet(m_settings, m_environment, q_set_size, m_settings->get_delta())
  , m_optimizer(optimizer)
  , m_randomGenerator(
    m_environment->action_properties().m_mean,
    m_environment->action_properties().m_standardDeviation
  )
  {
    //TODO: Duplicate network, and only update once a few iterations
  }

  void iterate(
    std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs, 
    std::optional<DataView> initial_state = {}
  ){
    std::vector<DataType> xp_states;
    std::vector<DataType> xp_actions;

    std::uint32_t iteration = 0;
    bool terminal = false;
    if(initial_state.has_value()){
      if(initial_state.value().size() != m_environment->state_size())
        throw std::runtime_error("Initial state has a different size, than what's expected by the environment");
      xp_states.push_back(initial_state.value().acquire());
    }else if(0 < m_qSet.get_number_of_sequences()){
      xp_states.push_back(m_qSet.get_input_sample(rand()%(m_qSet.get_number_of_sequences())));
    }else{ /* No initial state provided and nothing is in the q set, soo.. */
      //TODO: add random values?
      xp_states.push_back(DataType(m_environment->state_size(), 0.0));
    }
    xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
    while(terminal && (iteration < max_discovery_length)){
      auto next_state = m_environment->next(xp_states.back(), xp_actions.back());
      terminal = next_state.m_terminal;
      xp_actions.back() = XPSlot::action_slot(xp_actions.back(), next_state.m_resultQValue);
      if(!terminal && next_state.m_resultState.has_value()){
        xp_states.push_back(next_state.m_resultState.value());
        xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
      }
      ++iteration;
    }

    std::uint32_t q_set_size = m_qSet.get_number_of_sequences();
    m_qSet.incorporate(xp_states, xp_actions);
    if((0 == q_set_size) || (q_set_size != m_qSet.get_number_of_sequences()) ){
      m_optimizer->build(m_qSet, m_objective);
    }
    for(std::uint32_t training_iteration = 0; training_iteration < q_set_training_epochs; ++training_iteration){
      m_optimizer->iterate(m_qSet, (0 == training_iteration)/*force_gpu_upload*/);
    }
    ++m_iteration;
  }

private:
  inline static std::mt19937 s_randomEngine{std::random_device()};
  rafko_net::RafkoNet& m_network;
  rafko_net::SolutionSolver::Factory m_solverFactory;
  rafko_net::SolutionSolver m_agent;

  std::shared_ptr<RafQEnvironment> m_environment;
  std::shared_ptr<RafkoObjective> m_objective;
  RafQSet<ActionCount> m_qSet;
  std::shared_ptr<rafko_gym::RafkoAutodiffOptimizer> m_optimizer;
  std::normal_distribution<double> m_randomGenerator;
  std::uint32_t m_iteration = 0;

  DataType generate_action(DataView state, double exploration_ratio){
    MaybeDataType found_state;
    std::uint32_t found_state_index;
    if(
      (100 * exploration_ratio) < (rand()%100)
      ||(!(found_state = m_qSet.look_up(state, &found_state_index)).has_value())
    ){ /* Explore! Add a Random action */
      std::uint32_t action_item_index = 0u;
      std::uint32_t random_action_count = 0u;
      RafQSetItemConstView<ActionCount> action_for_state(
        m_qSet.get_input_sample(found_state_index),
        m_qSet.get_label_sample(found_state_index),
        m_environment->action_size()
      );
      DataType ret;
      std::generate_n(
        ret.begin(), m_environment->action_size(), 
        [this, &action_for_state, &action_item_index, &random_action_count, exploration_ratio]()->double{ 
          ++action_item_index;
          if((100 * exploration_ratio) < (rand()%100)){
            ++random_action_count;
            return m_randomGenerator(s_randomEngine); 
          }
          return action_for_state[action_item_index - 1];
        }
      );
      if(0 == random_action_count)
        ret[rand()%ret.size()] = m_randomGenerator(s_randomEngine);
      return ret;
    }else{ /* Exploit! Get best action for result state */
      return RafQSetItemConstView<ActionCount>(
        m_qSet.get_input_sample(found_state_index),
        m_qSet.get_label_sample(found_state_index),
        m_environment->action_size()
      )[0]; /*!Note: first action is the best one */
    }
  }
};

} /* namespace rafko_gym */

#endif /* RAFQ_TRAINER_H */
