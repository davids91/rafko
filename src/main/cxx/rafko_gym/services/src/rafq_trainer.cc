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
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

#include <algorithm>

namespace rafko_gym{

RafQTrainer::RafQTrainer(
  rafko_net::RafkoNet& network, std::uint32_t action_count, std::uint32_t q_set_size, 
  std::shared_ptr<RafQEnvironment> environment, Cost_functions cost_function,
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings
)
: RafQTrainer(
  network, std::make_shared<RafQSet>(*settings, *environment, action_count, q_set_size, settings->get_delta()), 
  environment, std::make_shared<RafkoCost>(*settings, cost_function), settings
)
{
  RFASSERT(static_cast<bool>(settings));
}

RafQTrainer::RafQTrainer(
  rafko_net::RafkoNet& network, std::uint32_t action_count, std::uint32_t q_set_size, 
  std::shared_ptr<RafQEnvironment> environment, std::shared_ptr<rafko_gym::RafkoObjective> objective,
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings
)
: RafQTrainer(
  network, std::make_shared<RafQSet>(*settings, *environment, action_count, q_set_size, settings->get_delta()),
  environment, objective, settings
)
{
  RFASSERT(static_cast<bool>(settings));
}

RafQTrainer::RafQTrainer(
  rafko_net::RafkoNet& network, std::shared_ptr<RafQSet> q_set, 
  std::shared_ptr<RafQEnvironment> environment, Cost_functions cost_function,
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings
)
: RafQTrainer(network, q_set, environment, std::make_shared<RafkoCost>(*settings, cost_function), settings)
{
  RFASSERT(static_cast<bool>(settings));
}

RafQTrainer::RafQTrainer(
  rafko_net::RafkoNet& network, std::shared_ptr<RafQSet> q_set, 
  std::shared_ptr<RafQEnvironment> environment, std::shared_ptr<rafko_gym::RafkoObjective> objective, 
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings
)
: RafkoAutonomousEntity(settings)
, m_stableNetwork(network)
, m_volatileNetwork(google::protobuf::Arena::Create<rafko_net::RafkoNet>(
  m_settings->get_arena_ptr(), m_stableNetwork
))
, m_environment(environment)
, m_objective(objective)
, m_qSet(q_set)
#if(RAFKO_USES_OPENCL)
, m_context(
  rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
    .build<rafko_mainframe::RafkoGPUContext>(*m_volatileNetwork, settings, m_objective)
)
, m_optimizer(
  rafko_mainframe::RafkoOCLFactory().select_platform().select_device()
    .build<RafkoAutodiffGPUOptimizer>(settings, *m_volatileNetwork, m_qSet, m_context)
)
#else
, m_context(std::make_shared<rafko_mainframe::RafkoCPUContext>(*m_volatileNetwork, settings, objective))
, m_optimizer(std::make_shared<RafkoAutodiffOptimizer>(settings, *m_volatileNetwork, m_context))
#endif/*(RAFKO_USES_OPENCL)*/
, m_randomActionGenerator(m_environment->action_properties().m_mean, m_environment->action_properties().m_standardDeviation)
{
  RFASSERT(static_cast<bool>(environment));
  RFASSERT(static_cast<bool>(q_set));
}

const RafQSet& RafQTrainer::q_set(){
  RFASSERT(static_cast<bool>(m_qSet));
  return *m_qSet;
}

void RafQTrainer::set_weight_updater(rafko_gym::Weight_updaters updater){
  RFASSERT(static_cast<bool>(m_optimizer));
  m_optimizer->set_weight_updater(updater);
}

void RafQTrainer::iterate(
  std::uint32_t max_discovery_length, double exploration_ratio, std::uint32_t q_set_training_epochs,
  const std::function<void(double/*progress*/, std::uint32_t/*step*/)>& progress_callback
){
  RFASSERT_SCOPE(RAFQ_ITERATION);
  RFASSERT_LOG("RafQ Iteration {}", m_iteration);
  std::vector<RafQEnvironment::AnyData> xp_user_data;
  std::vector<FeatureVector> xp_states;
  std::vector<FeatureVector> xp_actions;
  std::uint32_t discovery_iteration = 0;
  double q_set_iterations = m_qSet->get_number_of_sequences();
  double all_iterations = (max_discovery_length + q_set_training_epochs + q_set_iterations);
  double done_iterations = 0.0;
  RFASSERT_LOG("Estimated q-learning iterations: {}", all_iterations);
  progress_callback(0,0);
  if(0 < max_discovery_length){
    bool terminal = false;
    RFASSERT(m_environment->current_state().m_resultState.has_value());
    RafQEnvironment::StateTransition current_state = m_environment->current_state();
    xp_states.push_back(current_state.m_resultState.value().get());
    xp_user_data.push_back(std::move(current_state.m_userData));
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

      RafQEnvironment::StateTransition next_state = m_environment->next(xp_actions.back());
      terminal = next_state.m_terminal && next_state.m_resultState.has_value();
      xp_actions.back() = RafQSetItemConstView::action_slot(xp_actions.back(), next_state.m_resultQValue);
      RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
      if(!terminal && next_state.m_resultState.has_value() && (discovery_iteration < (max_discovery_length - 1))){
        xp_states.push_back(next_state.m_resultState.value());
        xp_user_data.push_back(std::move(next_state.m_userData));
        RFASSERT(xp_states.back().size() == m_environment->state_size());
        xp_actions.push_back(generate_action(xp_states.back(), exploration_ratio));
        RFASSERT(xp_actions.back().size() == m_environment->action_size());
      }else break;

      done_iterations = ++discovery_iteration;
      progress_callback(done_iterations / all_iterations, 1);
    }
    RFASSERT(xp_actions.back().size() == RafQSetItemConstView::feature_size(m_environment->action_size(), 1));
  }

  q_set_iterations = std::max(static_cast<std::size_t>(m_qSet->get_number_of_sequences()), xp_states.size());
  all_iterations = (max_discovery_length + q_set_training_epochs + q_set_iterations);
  RFASSERT_LOG("Q-learning iterations corrected: {}/{}", done_iterations, all_iterations);
  if((0 < xp_states.size()) && (0 < xp_actions.size())){
    m_qSet->incorporate(
      xp_states, xp_actions, std::move(xp_user_data), [&progress_callback, &done_iterations, &q_set_iterations, &all_iterations](double progress){ 
        done_iterations += 1.0;
        progress_callback((done_iterations + progress * q_set_iterations) / all_iterations, 2); 
      }
    );
  }

  progress_callback(done_iterations / all_iterations, 3);
  if(0 < q_set_training_epochs) /* Needs to build optimizer when the qset is the same size, because there might be changes within each state */
    m_optimizer->build(m_qSet, m_objective);
  for(std::uint32_t training_iteration = 0; training_iteration < q_set_training_epochs; ++training_iteration){
    m_optimizer->iterate(*m_qSet, (0 == training_iteration)/*force_gpu_upload*/);
    done_iterations += 1.0;
    progress_callback(done_iterations / all_iterations, 4);
  }

  if(0 == (m_iteration % m_settings->get_training_relevant_loop_count())){
    progress_callback(done_iterations / all_iterations, 5);
    RFASSERT_LOG("Updating weights of stable network..");
    m_stableNetwork.mutable_weight_table()->Assign(m_volatileNetwork->weight_table().begin(), m_volatileNetwork->weight_table().end());
    m_context->refresh_solution_weights();
  }
  progress_callback(1.0, 6);
  ++m_iteration;
}

RafQTrainer::FeatureVector RafQTrainer::generate_action(const FeatureVector& state, double exploration_ratio){
  static std::random_device rd{};
  static std::mt19937 gen{rd()};
  auto policy_actions = m_context->solve(state, true/*reset_neuron_data*/); /* The context is using the stable network */
  FeatureVector action_for_state = {policy_actions.begin(), policy_actions.begin() + m_environment->action_size()};

  RFASSERT_LOGV(action_for_state, "Action generated by policy:");
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
    RFASSERT_LOG("Modified {} elements..", random_action_count);
    if(0 == random_action_count)
      action_for_state[rand()%action_for_state.size()] = m_randomActionGenerator(gen);    
  } /* else --> Exploit! Get best action for current state */
  RFASSERT_LOGV(action_for_state, "Action generated by trainer:");
  return action_for_state;
}

} /* namespace rafko_gym */
