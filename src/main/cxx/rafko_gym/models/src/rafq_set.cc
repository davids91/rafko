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

//TODO: implement look_up function in openCL also?
RafQSet::MaybeFeatureVector RafQSet::look_up(FeatureView state, std::uint32_t* result_index_buffer) const{
  RFASSERT(state.size() == m_environment.state_size());
  MaybeFeatureVector result;
  const std::uint32_t item_count = get_number_of_sequences();
  const std::uint32_t items_in_one_thread = 1u + (item_count / m_lookupThreads.get_number_of_threads());
  m_lookupThreads.start_and_block([this, &state, &result, &result_index_buffer, item_count, items_in_one_thread](std::uint32_t thread_index){
    static std::atomic_bool someone_found_it = false;
    static std::mutex output_mutex; //TODO: Maybe move to members? 
    const std::uint32_t items_start_index = thread_index * items_in_one_thread;
    const std::uint32_t items_in_this_thread = std::min( items_in_one_thread, (item_count - std::min(item_count, items_start_index)) );
    for(std::uint32_t item_index = items_start_index; item_index < (items_start_index + items_in_this_thread); ++item_index){
      /*!Note: this is needed here in case any insertion inside the states buffer would cause a race condition */
      if(
        (!someone_found_it) /* If there are multiple matches, there might be interference */
        &&(m_costFunction.get_feature_error(state, get_input_sample(item_index), m_environment.state_size()) <= m_settings.get_delta())
      ){
        std::lock_guard<std::mutex> my_lock(output_mutex);
        if(!someone_found_it){
          result.emplace(get_input_sample(item_index));
          if(result_index_buffer)
            *result_index_buffer = item_index;
          someone_found_it = true;          
        }
      }
      if(someone_found_it)break;
    }

    std::lock_guard<std::mutex> my_lock(output_mutex);
    someone_found_it = false; /* restore state for the next run */
  });
  return result;
}

void RafQSet::incorporate(const std::vector<FeatureVector>& state_buffer, const std::vector<FeatureVector>& actions_buffer){
  // std::cout << "======================INCORPORATE STARTED======================"  << std::endl;
  // std::cout << "QSet: " << std::endl;
  // for(int i = 0; i < get_number_of_sequences(); ++i){
  //   const RafQSetItemConstView actions_view((*this)[i]);
  //   for(double d : get_input_sample(i)){
  //     std::cout << "[" << d << "]";
  //   }  
  //   std::cout << " == >  {";
  //   for(int j = 0; j < m_actionCount; ++j)
  //     std::cout << "(" << actions_view[j][0]<< ", q:" << actions_view.q_value(j) << ")";
  //   std::cout << "}" << std::endl;
  // }
  // std::cout << std::endl;
  RFASSERT(state_buffer.size() == actions_buffer.size());
  m_statesBuffer.reserve(m_statesBuffer.size() + state_buffer.size()); /* Reserve enough space so iteration invalidation can be minimized.. */
  m_actionsBuffer.reserve(m_actionsBuffer.size() + actions_buffer.size()); /* ..despite the slim possibility of actually filling up the reserved space */
  for(std::uint32_t state_index = 0; state_index < state_buffer.size(); ++state_index){
    // std::cout << "Incorporating vector of " << state_buffer.size() << std::endl;
    RFASSERT(state_buffer[state_index].size() == get_input_size());
    RFASSERT(actions_buffer[state_index].size() == RafQSetItemView::feature_size(m_environment.action_size(), 1u/*action count*/));
    std::uint32_t match_index;
    MaybeFeatureVector state_match = look_up(state_buffer[state_index], &match_index);
    const std::uint32_t action_size = m_environment.action_size();
    const RafQSetItemConstView new_action_view(state_buffer[state_index], actions_buffer[state_index], action_size, 1u /*action count*/);
    const double new_action_q_value = new_action_view.q_value() + get_td_value(new_action_view, new_action_view.q_value());
    // std::cout << "Looking for state " << state_buffer[state_index][0] << std::endl;
    if(state_match.has_value()){
      // std::cout << "found state (" << state_buffer[state_index][0] << ") in the set!" << std::endl;
      RFASSERT(match_index < get_number_of_sequences());
      std::uint32_t action_index = m_actionCount;
      RafQSetItemView stored_action_view((*this)[match_index]);
      for(action_index = 0; action_index < m_actionCount; ++action_index){
        // std::cout << "comparing actions: " << stored_action_view[action_index][0] << " <> " << new_action_view[0][0] << std::endl;
        if( m_settings.get_delta_2() >= m_costFunction.get_feature_error(
          {stored_action_view[action_index], action_size}, {new_action_view[0], action_size}, action_size
        ) ) break; /* if the difference is small enough, a match is found! */
      }
      if(action_index < m_actionCount){ /* Update the QValue based on TD Learning */
        // std::cout << " found action[" << action_index << "]" << std::endl;
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
        // std::cout << "Did not find a matching action!" << std::endl;
        std::uint32_t action_index = m_actionCount;
        do{ /* Find the index of the new action */
          --action_index;
          // std::cout << "comparing action[" << action_index << "] qvalues: " 
          // << "(" << stored_action_view.q_value(action_index) << " * (1.0 + " << m_overwriteQThreshold << "))"
          // << " > (" << new_action_view.q_value() << " or " << new_action_q_value << ")"
          // << std::endl;
          if(new_action_q_value < (stored_action_view.q_value(action_index))){
            // --> there is an error because the negative q value is not at the end of the actions ?! 
            // std::cout << "comparing q values: " << new_action_q_value 
            // << " <> action[" << action_index << "]:" 
            // << stored_action_view.q_value(action_index) 
            // << "; min: " << stored_action_view.min_q_value()
            // << "; thr: " << m_overwriteQThreshold << ";" 
            // << std::endl;
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
        // std::cout << "Overwriting action[" << action_index << "]!" << std::endl;
        /*!Note: there will surely be match, because the action because of the entry condition of this block */
        std::uint32_t i = m_actionCount; /* write over the actions starting from the worst */
        do{
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
      // std::cout << "Copying Action value: " << new_action_view[0][0] << "; q value "  << new_action_q_value 
      // << "to index " << target_action_index
      // << std::endl;
      std::copy( 
        new_action_view[0], new_action_view[0] + m_environment.action_size(),
        RafQSetItemView::action_iterator(m_actionsBuffer.back(), m_environment.action_size(), target_action_index)
      ); 
      *RafQSetItemView::q_value_iterator(m_actionsBuffer.back(), m_environment.action_size(), target_action_index) = new_action_q_value;
      /*!Note: Since this state at this point only has 1 action, it is the best one; which needs to be the first of the 
       * actions.
       */
    }
  }/*for(every incoming state-action pair)*/
  keep_best(m_maxSetSize);
  // std::cout << "======================INCORPORATE FINISHED======================"  << std::endl;
  // std::cout << "QSet: " << std::endl;
  // for(int i = 0; i < get_number_of_sequences(); ++i){
  //   const RafQSetItemConstView actions_view((*this)[i]);
  //   for(double d : get_input_sample(i)){
  //     std::cout << "[" << d << "]";
  //   }  
  //   std::cout << " == >  {";
  //   for(int j = 0; j < m_actionCount; ++j)
  //     std::cout << "(" << actions_view[j][0]<< ", q:" << actions_view.q_value(j) << ")";
  //   std::cout << "}" << std::endl;
  // }
  // // std::cout << std::endl << "From: " << std::endl;
  // // for(int i = 0; i < state_buffer.size(); ++i){
  // //   for(double d : state_buffer[i]){
  // //     std::cout << "[" << d << "]";
  // //   }   
  // //   std::cout << "--> ";
  // //   for(double d : actions_buffer[i]){
  // //     std::cout << "[" << d << "]";
  // //   }   
  // //   std::cout << std::endl;
  // // }
  // std::cout << std::endl;
}

void RafQSet::erase_worst(std::uint32_t count){
  // std::cout << "======================"  << std::endl;
  // std::cout << "QSet: " << std::endl;
  // for(int i = 0; i < get_number_of_sequences(); ++i){
  //   const RafQSetItemConstView actions_view((*this)[i]);
  //   std::cout << get_input_sample(i)[0] << " == >  {";
  //   for(int j = 0; j < m_actionCount; ++j)
  //     std::cout << "(" << actions_view[j][0]<< ", q:" << actions_view.q_value(j) << ")";
  //   std::cout << "}" << std::endl;
  // }
  // std::cout << std::endl;
  // std::cout << "Erasing worst " << count << " elements.." << std::endl;
  RFASSERT(count < get_number_of_sequences());
  std::map<double, std::uint32_t, std::greater<double>> worst_q_values;
  double best_in_worst_q_value = std::numeric_limits<double>::max();
  for(std::uint32_t item_index = 0; item_index < get_number_of_sequences(); item_index++){
    // std::cout << "checking item[" << item_index << "]..." << std::endl;
    RafQSetItemView item_view((*this)[item_index]);
    double current_q_value = item_view.avg_q_value();
    if(current_q_value < best_in_worst_q_value){
      worst_q_values.insert({current_q_value, item_index});
      RafQSetItemView view((*this)[worst_q_values.begin()->second]);
      // std::cout << "Worst items in set: ";
      // for(const auto& [q_value, index] : worst_q_values){
      //   RafQSetItemView view((*this)[index]);
      //   std::cout << "{" 
      //   << view.state()[0]<< "," << view[0][0] 
      //   << "(q value: " << q_value << ")"
      //   << "},";
      // }
      // std::cout << std::endl;

      if(worst_q_values.size() > count)
        worst_q_values.erase(worst_q_values.begin());
    }
  }/*for(every item in the set)*/
  // std::cout << "Finally.. Worst items in set: ";
  // for(const auto& [q_value, index] : worst_q_values){
  //   RafQSetItemView view((*this)[index]);
  //   std::cout << "{" 
  //   << view.state()[0]<< "," << view[0][0] 
  //   << "(q value: " << q_value << ")"
  //   << "},";
  // }
  // std::cout << std::endl;
  std::map<std::uint32_t, double, std::greater<std::uint32_t>> to_delete;
  for(auto& [q_value, index] : worst_q_values){
    to_delete.insert({index, q_value});
  }
  for(auto& [index, q_value] : to_delete){
    m_statesBuffer.erase(m_statesBuffer.begin() + index);
    m_actionsBuffer.erase(m_actionsBuffer.begin() + index);
  }
}

//TODO: Experiment: each look ahead adds the delta ( improvement) of q-values
//TODO: Td policy: for creating td; essentially a function for temporal q-value sequence
double RafQSet::get_td_value(const RafQSetItemConstView& new_action_view, double old_q_value) const{
  // std::cout << "\n\n calculating td value.." << std::endl;
  double temporal_difference_value = new_action_view.q_value(0); /* Reqward: the only Q-value in @new_action_view */
  if(0 < m_settings.get_look_ahead_count()){
    double lambda = m_settings.get_gamma();
    std::uint32_t next_state_index;
    MaybeFeatureVector next_state_data(new_action_view.state());
    RafQSetItemConstView next_state_view(new_action_view);
    // std::cout << "calculating q value; starting q value:" << temporal_difference_value << std::endl;
    for(std::uint16_t look_ahead_index = 0; look_ahead_index < m_settings.get_look_ahead_count(); ++look_ahead_index){
      RFASSERT(next_state_data.has_value());
      RafQEnvironment::StateTransition state_transition;
      if(0 == look_ahead_index){ /* In the first look ahead iteration the current action runs instead of the best */
        // std::cout << "//q value " << new_action_view.max_q_value() << " -->future[" << look_ahead_index << "]:"
        // // << "{" << next_state_view.state()[0] << ","
        // // << new_action_view[0] << "}"
        // << std::endl;
        new (&state_transition) RafQEnvironment::StateTransition(m_environment.next(
          {next_state_view.state().begin(), next_state_view.state().end()}, {next_state_view[0], m_environment.action_size()}
        ));
      }else{
        RFASSERT(next_state_index < get_number_of_sequences());
        // std::cout << "//q value " << next_state_view.max_q_value() << " -->future[" << look_ahead_index << "]:"
        // // << "{" << next_state_view.state()[0] << ","
        // // << next_state_view[0][0] << "(best action)}"
        // << std::endl; 
        new (&state_transition) RafQEnvironment::StateTransition(m_environment.next(
          {next_state_view.state().begin(), next_state_view.state().end()},
          {next_state_view[0], next_state_view[0] + m_environment.action_size()}
        )); /*!Note: The first action also has the highest q-value */
      }

      if(!state_transition.m_resultState.has_value()){
        // std::cout << "No further state transitions stored!" << std::endl;
        break;
      }

      next_state_data = look_up(state_transition.m_resultState.value().get(), &next_state_index);
      if(next_state_data.has_value()){
        RFASSERT(next_state_index < get_number_of_sequences());
        new (&next_state_view) RafQSetItemConstView((*this)[next_state_index]);
        // std::cout << "resulting next state: " 
        // << "q value " << next_state_view.max_q_value() << " or maybe " << next_state_view.q_value(0) << " -->"
        // << "{" << next_state_view.state()[0] << "," 
        // << next_state_view[0][0] << "(best action)}"
        // << std::endl;
        // std::cout << " + (" << lambda << " * " << next_state_view.max_q_value() << ")" << std::endl;
        temporal_difference_value += lambda * next_state_view.max_q_value();
        lambda = std::pow(lambda, 2.0);              
      }else{
        // std::cout << "State " << state_transition.m_resultState.value().get()[0] << " Not found in set!" << std::endl;
        break;
      }

      if(state_transition.m_terminal){
        // std::cout << "state is terminal.." << std::endl;
        break;
      }
    }
  }
  return (temporal_difference_value - old_q_value) * m_settings.get_learning_rate();
}

} /* namespace rafko_gym */
