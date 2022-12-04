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
#include "rafko_gym/services/rafq_trainer.hpp"

#include <algorithm>

namespace rafko_gym{

void RafQTrainer::iterate(
  std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs,
  const std::function<void(double/*progress*/)>& progress_callback
){
  std::vector<FeatureVector> xp_states;
  std::vector<FeatureVector> xp_actions;
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
        xp_states.push_back(next_state.m_resultState.value());
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

RafQTrainer::FeatureVector RafQTrainer::generate_action(const FeatureVector& state, double exploration_ratio){
  static std::random_device rd{};
  static std::mt19937 gen{rd()};
  auto policy_actions = m_context->solve(state, true/*reset_neuron_data*/); /* The context is using the stable network */
  FeatureVector action_for_state = {policy_actions.begin(), policy_actions.begin() + m_environment->action_size()};

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

} /* namespace rafko_gym */
