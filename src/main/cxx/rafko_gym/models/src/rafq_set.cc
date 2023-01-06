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
#include "rafko_gym/models/rafq_set.hpp"

#include <algorithm>
#include <limits>
#include <atomic>

namespace rafko_gym{

DataSetPackage RafQSet::generate_best_sequences(std::uint32_t preferred_sequence_size) const{
  RFASSERT_LOG("Generating best action sequences..");
  std::size_t max_sequence_length = 0;
  std::vector<std::vector<std::size_t>> index_sequences;
  std::vector<bool> included(get_number_of_sequences(), false);
  std::size_t start_in_set = 0u;
  RafQEnvironment::StateTransition state_transition{{}, 0.0, false};
  while(start_in_set < get_number_of_sequences()){
    RFASSERT_LOG("Start of new sequence: {}", start_in_set);
    if(!included[start_in_set]){
      index_sequences.push_back({});
      std::uint32_t next_state_index = start_in_set;
      MaybeFeatureVector next_state_data(get_input_sample(next_state_index));
      while(next_state_data.has_value() && (index_sequences.back().size() < m_maxSetSize)){
        RFASSERT(next_state_index < get_number_of_sequences());
        index_sequences.back().push_back(next_state_index);
        included[next_state_index] = true;
        RFASSERT_LOG(
          "including state[{}] to super-sequence[{}]; transition: q: {}; terminal: {}", 
          next_state_index, (index_sequences.size()-1), 
          state_transition.m_resultQValue, state_transition.m_terminal
        );
        if(state_transition.m_terminal)
          break; /* previous state was terminal, don't look for another state */

        new (&state_transition) RafQEnvironment::StateTransition(
          m_environment.next(
            next_state_data.value().get(), 
            RafQSetItemConstView::best_action_slot(
              get_label_sample(next_state_index), m_environment.action_size()
            )
          )
        );

        if(state_transition.m_resultState.has_value())
          next_state_data = look_up(state_transition.m_resultState.value().get(), &next_state_index);
      }
      if(max_sequence_length < index_sequences.back().size())
        max_sequence_length = index_sequences.back().size();
    }/* if( current index is not yet included in set ) */
    ++start_in_set;
  }/* while( current index in set ) */

  DataSetPackage result;
  result.set_input_size(get_input_size());
  result.set_feature_size(m_environment.action_size());
  result.set_sequence_size(preferred_sequence_size);

  RFASSERT_LOG("Maximum length of sequences: {} >= preferred sequence size: {}", max_sequence_length, preferred_sequence_size);
  if(max_sequence_length >= preferred_sequence_size){
    for(const std::vector<std::size_t>& index_sequence : index_sequences){
      if(index_sequence.size() < preferred_sequence_size)
        continue;

      std::size_t sequence_start_index = 0;
      while(sequence_start_index < index_sequence.size()){
        const std::size_t actual_sequence_start = (
          sequence_start_index - (
            preferred_sequence_size - std::min(
              static_cast<std::size_t>(preferred_sequence_size), (index_sequence.size() - sequence_start_index)
            )
          )
        );
        for(std::size_t index = actual_sequence_start; index < (actual_sequence_start + preferred_sequence_size); ++index){
          const std::size_t i = index_sequence[index];
          FeatureView action_slot = RafQSetItemConstView::best_action_slot(get_label_sample(i), m_environment.action_size());
          result.mutable_inputs()->Add(get_input_sample(i).begin(), get_input_sample(i).end());
          result.mutable_labels()->Add(action_slot.begin(), action_slot.end());
        }
        sequence_start_index += preferred_sequence_size;
      }
    }
  }

  return result;
}


RafQSet::MaybeFeatureVector RafQSet::look_up(FeatureView state, std::uint32_t* result_index_buffer) const{
  RFASSERT_LOGV(state.acquire(), "Looking for state: ");
  RFASSERT(state.size() == m_environment.state_size());
  MaybeFeatureVector result;
  const std::uint32_t item_count = get_number_of_sequences();
  const std::uint32_t items_in_one_thread = 1u + (item_count / m_lookupThreads.get_number_of_threads());
  m_lookupThreads.start_and_block([this, &state, &result, &result_index_buffer, item_count, items_in_one_thread](std::uint32_t thread_index){
    static std::atomic_bool someone_found_it = false;
    const std::uint32_t items_start_index = thread_index * items_in_one_thread;
    const std::uint32_t items_in_this_thread = std::min( items_in_one_thread, (item_count - std::min(item_count, items_start_index)) );
    for(std::uint32_t item_index = items_start_index; item_index < (items_start_index + items_in_this_thread); ++item_index){
      /*!Note: this is needed here in case any insertion inside the states buffer would cause a race condition */
      if(
        (!someone_found_it) /* If there are multiple matches, there might be interference */
        &&(m_costFunction.get_feature_error(state, get_input_sample(item_index), m_environment.state_size()) <= m_settings.get_delta())
      ){
        std::lock_guard<std::mutex> my_lock(m_searchResultMutex);
        if(!someone_found_it){
          result.emplace(get_input_sample(item_index));
          if(result_index_buffer)
            *result_index_buffer = item_index;
          someone_found_it = true;          
        }
      }
      if(someone_found_it)break;
    }

    if(someone_found_it){
      std::lock_guard<std::mutex> my_lock(m_searchResultMutex);
      someone_found_it = false; /* restore state for the next run */      
    }
  });
  RFASSERT_LOG("Result value is {}", (result.has_value())?"set!":"not set!");
  return result;
}

void RafQSet::incorporate(
  const std::vector<FeatureVector>& state_buffer, const std::vector<FeatureVector>& actions_buffer, 
  const std::function<void(double/*progress*/)>& progress_callback
){
  RFASSERT_SCOPE(QSET_INCORPORATE)
  RFASSERT_LOG("Incorporating {} states and {} actions to q-set!", state_buffer.size(), actions_buffer.size());
  RFASSERT(state_buffer.size() == actions_buffer.size());
  m_statesBuffer.reserve(m_statesBuffer.size() + state_buffer.size()); /* Reserve enough space so iteration invalidation can be minimized.. */
  m_actionsBuffer.reserve(m_actionsBuffer.size() + actions_buffer.size()); /* ..despite the slim possibility of actually filling up the reserved space */
  for(std::uint32_t state_index = 0; state_index < state_buffer.size(); ++state_index){
    RFASSERT_LOG("state[{}] size: {}", state_index, state_buffer[state_index].size());
    RFASSERT(state_buffer[state_index].size() == get_input_size());
    RFASSERT(actions_buffer[state_index].size() == RafQSetItemView::feature_size(m_environment.action_size(), 1u/*action count*/));
    std::uint32_t match_index;
    MaybeFeatureVector state_match = look_up(state_buffer[state_index], &match_index);
    const std::uint32_t action_size = m_environment.action_size();
    const RafQSetItemConstView new_action_view(state_buffer[state_index], actions_buffer[state_index], action_size, 1u /*action count*/);
    const double new_action_q_value = new_action_view.q_value() + get_td_value(new_action_view, new_action_view.q_value());
    if(state_match.has_value()){
      RFASSERT(match_index < get_number_of_sequences());
      std::uint32_t action_index = m_actionCount;
      RafQSetItemView stored_action_view((*this)[match_index]);
      FeatureView stored_action_vector_view(stored_action_view[0], action_size);
      for(action_index = 0; action_index < m_actionCount; ++action_index){
        if(0 < action_index)
          new (&stored_action_vector_view) FeatureView(stored_action_view[action_index], action_size);
        if( m_settings.get_delta_2() >= m_costFunction.get_feature_error(
          stored_action_vector_view, {new_action_view[0], action_size}, action_size
        ) ) break; /* if the difference is small enough, a match is found! */
      }
      if(action_index < m_actionCount){ /* Update the QValue based on TD Learning */
        RFASSERT_LOGV(stored_action_vector_view.acquire(), "found action[{}]: ", action_index);
        const double new_q_value = stored_action_view.q_value(action_index) + get_td_value(new_action_view, stored_action_view.q_value(action_index));
        stored_action_view.set_q_value(new_q_value, action_index);

        /* Updated q-value may have modified the order of possibly multiple actions */
        while(action_index < (m_actionCount - 1)){ /* Swap with better actions in worse places */
          if(stored_action_view.q_value(action_index + 1) > new_q_value)
            stored_action_view.swap_action(action_index + 1, action_index);
            else break;
          ++action_index;
        }
        while(action_index > 0u){ /* Swap with worse actions in better places */
          if(stored_action_view.q_value(action_index - 1) < new_q_value)
            stored_action_view.swap_action(action_index - 1, action_index);
            else break;
          --action_index;
        }
      }else if( /* state is present, but the action is new. Take it over in case the qvalue is greate4r, than the worse */
        ( /* In case the q value is positive, the percentage is added to 1 */
          (0 <= new_action_q_value) /* so the bigger value is being comapred to the new action */
          && (new_action_q_value > (stored_action_view.min_q_value() * (1.0 + m_overwriteQThreshold)))
        )||( /* In case the q value is negative, the percentage is substraceted from 1 */
          (0 > new_action_q_value) /* so the bigger value is being comapred to the new action in this case as well */
          && (new_action_q_value > (stored_action_view.min_q_value() * (1.0 - m_overwriteQThreshold)))
        )
      ){
        RFASSERT_LOG(
          "Did not find a matching action, but q-value {} is higher, than {} * {}!",
          new_action_q_value, stored_action_view.min_q_value(), m_overwriteQThreshold
        );
        std::uint32_t action_index = m_actionCount;

        do{ /* Find the index of the new action */
          --action_index;
          if(1 == m_actionCount) break; /* no need to look for anything, since there's only one action! */
          RFASSERT_LOG(
            "comparing new action to stored action[{}], q_value: {}",
            action_index, stored_action_view.q_value(action_index)
          );
          if(new_action_q_value < (stored_action_view.q_value(action_index))){
            RFASSERT_LOG("stored action[{}] is better; won't overwrite it!", action_index);
            ++action_index;
            RFASSERT(action_index < m_actionCount);
            /*!Note: Because the entry point of the outer block had the condition, that the new q value is greater, than the minimum 
             * it should be guaranteed that the new q value is greater, than the last index, and by extension, the index must be 
             * smaller than m_actionCount here.
             * The reason for that is that the condition for this block is the current stored q value is greater, than the new,
             * which means that @action_index is at the index after its designated index.
             **/
            break;
          }
        }while(0 < action_index);
        RFASSERT_LOG("Overwriting stored action[{}]", action_index);
        /*!Note: there will surely be match, because the action because of the entry condition of this block */
        std::uint32_t i = m_actionCount; /* overwrite the actions starting from the worst */
        do{
          if(1 == m_actionCount) break; /* no need to do anything, since there's only one action! */
          --i;
          stored_action_view.copy_action(i-1/*source*/, i/*target*/);
        }while(i > (action_index + 1));
        stored_action_view.take_over(new_action_view, 0, action_index);
        stored_action_view.set_q_value(new_action_q_value, action_index);
      }
      m_avgQValue[match_index] = stored_action_view.avg_q_value();
    }else{ /* no match is found for the state, extend the database with the newly found state */
      m_statesBuffer.emplace_back(state_buffer[state_index]);
      m_actionsBuffer.emplace_back(get_feature_size());
      m_avgQValue.emplace_back(new_action_q_value);
      const std::uint32_t target_action_index = (0 <= new_action_q_value)?(0):(m_actionCount - 1);
      RFASSERT_LOGV(
        new_action_view.action().acquire(), "Copying new action with q-value {} to index[{}]:", 
        new_action_q_value, target_action_index
      );
      std::copy( 
        new_action_view[0], new_action_view[0] + m_environment.action_size(),
        RafQSetItemView::action_iterator(m_actionsBuffer.back(), m_environment.action_size(), target_action_index)
      ); 
      *RafQSetItemView::q_value_iterator(m_actionsBuffer.back(), m_environment.action_size(), target_action_index) = new_action_q_value;
      /*!Note: Since this state at this point only has 1 action, it is the best one; which needs to be the first of the 
       * actions.
       */
    }
    progress_callback( static_cast<double>(state_index) / static_cast<double>(
      state_buffer.size() + get_number_of_sequences() - std::min(get_number_of_sequences(), m_maxSetSize)
    ) );
  }/*for(every incoming state-action pair)*/
  RFASSERT_LOG("Resulting q-set size: {} / {}", get_number_of_sequences(), m_maxSetSize);
  keep_best(m_maxSetSize);
}

void RafQSet::erase_worst(std::uint32_t count){
  RFASSERT_LOG("Erasing worst {} elements from set of size {}", count, get_number_of_sequences());
  RFASSERT(count < get_number_of_sequences());
  std::map<double, std::uint32_t, std::greater<double>> worst_q_values;
  double best_in_worst_q_value = std::numeric_limits<double>::max();
  for(std::uint32_t item_index = 0; item_index < get_number_of_sequences(); item_index++){
    RFASSERT_LOG("Checking index {}", item_index);
    RafQSetItemView item_view((*this)[item_index]);
    double current_q_value = item_view.avg_q_value();
    if(current_q_value < best_in_worst_q_value){
      worst_q_values.insert({current_q_value, item_index});
      RafQSetItemView view((*this)[worst_q_values.begin()->second]);
      RFASSERT_LOG("Worst items in set:");
      for([[maybe_unused]]const auto& [q_value, index] : worst_q_values)
        RFASSERT_LOGV((*this)[index].action().acquire(),"index[{}], q-value {}:", index, q_value);      

      if(worst_q_values.size() > count)
        worst_q_values.erase(worst_q_values.begin());
    }
  }/*for(every item in the set)*/
  RFASSERT_LOG("..Finally worst items in set: ");
    for([[maybe_unused]]const auto& [q_value, index] : worst_q_values)
      RFASSERT_LOGV((*this)[index].action().acquire(),"index[{}], q-value {}:", index, q_value);      

  /* Erasing worst items */
  std::map<std::uint32_t, double, std::greater<std::uint32_t>> to_delete;
  for(auto& [q_value, index] : worst_q_values){
    to_delete.insert({index, q_value});
  }
  for(auto& [index, q_value] : to_delete){
    m_statesBuffer.erase(m_statesBuffer.begin() + index);
    m_actionsBuffer.erase(m_actionsBuffer.begin() + index);
  }
}

double RafQSet::get_td_value(const RafQSetItemConstView& new_action_view, double old_q_value) const{
  RFASSERT_LOG("Calculating temporal difference value, based on latest reward: {}", new_action_view.q_value(0));
  double temporal_difference_value = new_action_view.q_value(0); /* Reward: the only Q-value in @new_action_view */
  if(0 < m_settings.get_look_ahead_count()){
    RFASSERT_LOG("Looking {} loops ahead..", m_settings.get_look_ahead_count());
    double lambda = m_settings.get_gamma();
    std::uint32_t next_state_index;
    MaybeFeatureVector next_state_data(new_action_view.state());
    RafQSetItemConstView next_state_view(new_action_view);
    for(std::uint16_t look_ahead_index = 0; look_ahead_index < m_settings.get_look_ahead_count(); ++look_ahead_index){
      RFASSERT_LOG("future[{}]---", look_ahead_index);
      RFASSERT(next_state_data.has_value());
      RFASSERT_LOG("max q-value: {}", next_state_view.max_q_value());
      if(0 < look_ahead_index) /* In the first look ahead iteration the current action runs instead of the best */
        RFASSERT(next_state_index < get_number_of_sequences());
        else{
          RFASSERT_LOG("..of new action..");
        }
      RafQEnvironment::StateTransition state_transition = m_environment.next(
        next_state_view.state(), {next_state_view[0], m_environment.action_size()}
      ); /*!Note: The first action also has the highest q-value */

      if(!state_transition.m_resultState.has_value()){
        RFASSERT_LOG("Environment doesn't contain a next step..");
        break;
      }

      next_state_data = look_up(state_transition.m_resultState.value().get(), &next_state_index);
      if(next_state_data.has_value()){
        RFASSERT_LOG("New state found!");
        RFASSERT(next_state_index < get_number_of_sequences());
        new (&next_state_view) RafQSetItemConstView((*this)[next_state_index]);
        RFASSERT_LOG(
          "TD Value updated with: {} * {} ==> {}", 
          lambda, next_state_view.max_q_value(), 
          (temporal_difference_value + (lambda * next_state_view.max_q_value()))
        );
        temporal_difference_value += lambda * next_state_view.max_q_value();
        lambda = std::pow(lambda, 2.0);              
      }else{
        RFASSERT_LOG("Couldn't find new state in q-set!");
        break;
      }

      if(state_transition.m_terminal){
        RFASSERT_LOG("New state is terminal");
        break;
      }
    }
  }/*if(settings permit looking ahead)*/
  return (temporal_difference_value - old_q_value) * m_settings.get_learning_rate();
}

} /* namespace rafko_gym */
