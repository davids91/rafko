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
  using DataType = RafQEnvironment::DataType;
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
  , m_network(network)
  , m_environment(environment)
  , m_objective(objective)
  , m_qSet(std::make_shared<RafQSet>(
    *m_settings, *m_environment, action_count, q_set_size, m_settings->get_delta()
  ))
  #if(RAFKO_USES_OPENCL)
  , m_context(
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<rafko_mainframe::RafkoGPUContext>(network, settings, m_objective)
  )
  , m_optimizer(
    rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
      .build<RafkoAutodiffGPUOptimizer>(settings, network, m_qSet, m_context)
  )
  #else
  , m_context(std::make_shared<rafko_mainframe::RafkoCPUContext>(network, settings, objective))
  , m_optimizer(std::make_shared<RafkoAutodiffOptimizer>(settings, network, m_context))
  #endif/*(RAFKO_USES_OPENCL)*/
  , m_randomActionGenerator(m_environment->action_properties().m_mean, m_environment->action_properties().m_standardDeviation)
  {
    //TODO: Duplicate network, and only update once a few iterations
  }

  std::uint32_t q_set_size() const{
    return m_qSet->get_number_of_sequences();
  }

  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u, bool force_gpu_upload = false){
    return m_context->stochastic_evaluation(to_seed, seed_value, force_gpu_upload);
  }

  void iterate(std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs){
    std::vector<DataType> xp_states;
    std::vector<DataType> xp_actions;
    std::cout << "(" << __LINE__ << ")" << std::endl;
    if(0 < max_discovery_length){
      std::uint32_t iteration = 0;
      bool terminal = false;
      m_environment->reset();
      RFASSERT(m_environment->current_state().has_value());
      xp_states.push_back(m_environment->current_state().value().get());
      RFASSERT(xp_states.back().size() == m_environment->state_size());
      xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
      RFASSERT(xp_actions.back().size() == m_environment->action_size());
      [[maybe_unused]] std::uint32_t xp_states_size = 0;
      [[maybe_unused]] std::uint32_t xp_actions_size = 0;
      while(!terminal && (iteration < max_discovery_length)){
        RFASSERT(xp_states_size < xp_states.size());
        RFASSERT(xp_actions_size < xp_actions.size());
        xp_states_size = xp_states.size();
        xp_actions_size = xp_actions.size();

        auto next_state = m_environment->next(xp_actions.back());
        terminal = next_state.m_terminal && next_state.m_resultState.has_value();
        xp_actions.back() = RafQSetItemConstView::action_slot(xp_actions.back(), next_state.m_resultQValue);
        RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
        if(!terminal && next_state.m_resultState.has_value() && (iteration < (max_discovery_length - 1))){
          xp_states.push_back(next_state.m_resultState.value()); //TODO: bad alloc?!?!
          RFASSERT(xp_states.back().size() == m_environment->state_size());
          xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
          RFASSERT(xp_actions.back().size() == m_environment->action_size());
        }else break;
        ++iteration;
      }
      RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
    }
    std::cout << "(" << __LINE__ << ")" << std::endl;
    
    std::uint32_t q_set_size = m_qSet->get_number_of_sequences();
    m_qSet->incorporate(xp_states, xp_actions);
    std::cout << "q_set_size: " << q_set_size << "; m_qSet->get_number_of_sequences(): " << m_qSet->get_number_of_sequences() << std::endl;
    if((0 == q_set_size) || (q_set_size != m_qSet->get_number_of_sequences()) ){
      m_optimizer->build(m_qSet, m_objective);
    }
    std::cout << "(" << __LINE__ << ")" << std::endl;
    for(std::uint32_t training_iteration = 0; training_iteration < q_set_training_epochs; ++training_iteration){
      m_optimizer->iterate(*m_qSet, (0 == training_iteration)/*force_gpu_upload*/);
      std::cout << "\r(" << __LINE__ << ")" << training_iteration;
    }
    std::cout << "\n(" << __LINE__ << ")" << std::endl;
    ++m_iteration;
  }

private:
  rafko_net::RafkoNet& m_network;

  std::shared_ptr<RafQEnvironment> m_environment;
  std::shared_ptr<RafkoObjective> m_objective;
  std::shared_ptr<RafQSet> m_qSet;
  std::shared_ptr<rafko_mainframe::RafkoContext> m_context;
  std::shared_ptr<RafkoAutodiffOptimizer> m_optimizer;
  std::normal_distribution<double> m_randomActionGenerator;
  std::uint32_t m_iteration = 0;

  DataType generate_action(DataView state, double exploration_ratio){
    static std::random_device rd{};
    static std::mt19937 gen{rd()};
    MaybeDataType found_state;
    std::uint32_t found_state_index;
    if(
      (100 * exploration_ratio) < (rand()%100)
      ||(!(found_state = m_qSet->look_up(state, &found_state_index)).has_value())
    ){ /* Explore! Add a Random action */
      std::uint32_t action_item_index = 0u;
      std::uint32_t random_action_count = 0u;
      DataType ret(m_environment->action_size());
      if(found_state.has_value()){
        RafQSetItemConstView actions_for_state(
          m_qSet->get_input_sample(found_state_index), m_qSet->get_label_sample(found_state_index),
          m_qSet->action_count(), m_environment->action_size()
        );
        std::generate_n(
          ret.begin(), m_environment->action_size(), 
          [this, &actions_for_state, &action_item_index, &random_action_count, exploration_ratio]()->double{ 
            ++action_item_index;
            if((100 * exploration_ratio) < (rand()%100)){
              ++random_action_count;
              return m_randomActionGenerator(gen); 
            }
            return actions_for_state[0][action_item_index - 1]; /* @action_item_index'th item of the best ( 0th ) action */
          }
        );
        if(0 == random_action_count)
          ret[rand()%ret.size()] = m_randomActionGenerator(gen);
        // std::cout << "action size: " << ret.size() << std::endl;
        return ret;
      }else{
        std::generate_n(ret.begin(), m_environment->action_size(), [this]()->double{ return m_randomActionGenerator(gen); });
        // std::cout << "action size: " << ret.size() << std::endl;
        return ret;
      }
    }else{ /* Exploit! Get best action for result state */
      RafQSetItemConstView actions_for_state(
        m_qSet->get_input_sample(found_state_index), m_qSet->get_label_sample(found_state_index),
        m_environment->action_size(), m_qSet->action_count()
      );
      // std::cout << "action size: " << m_environment->action_size() << std::endl;
      return { actions_for_state[0], actions_for_state[0] + m_environment->action_size() }; /*!Note: first action is the best one */
    }
  }
};

} /* namespace rafko_gym */

#endif /* RAFQ_TRAINER_H */
