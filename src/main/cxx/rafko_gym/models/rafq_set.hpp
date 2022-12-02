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

#ifndef RAFQ_SET_H
#define RAFQ_SET_H

#include "rafko_global.hpp"

#include <vector>
#include <cassert>
#include <algorithm>
#include <functional>
#include <optional>
#include <map>
#include <limits>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/services/cost_function_mse.hpp"

namespace rafko_gym{

/**
 * @brief      This class helps query a state paired with a number of Action Q-value pairs set by its template arguments
 *             providing only const access
 */
class RAFKO_EXPORT RafQSetItemConstView
{
  static std::vector<std::uint32_t> q_value_offset(std::uint32_t  action_size, std::uint32_t action_count){
    std::vector<std::uint32_t> offsets;
    for(std::uint32_t action_index = 0u; action_index < action_count; ++action_index)
      offsets.push_back( action_index * (action_size + 1u) );
    return offsets;
  }

  static std::vector<std::uint32_t> action_offset(std::uint32_t  action_size, std::uint32_t action_count){
    std::vector<std::uint32_t> offsets;
    for(std::uint32_t action_index = 0u; action_index < action_count; ++action_index)
      offsets.push_back( action_index * (action_size + 1u) + 1u );
    return offsets;
  }

public:
  using DataType = RafQEnvironment::DataType;

  RafQSetItemConstView(
    const DataType& state, const DataType& actions,
    std::uint32_t action_size, std::uint32_t action_count = 1
  )
  : m_actionCount(action_count)
  , m_state(state)
  , m_actions(actions)
  , m_stateSize(state.size())
  , m_actionSize(action_size)
  , m_qValueOffset(q_value_offset(m_actionSize, m_actionCount))
  , m_actionOffset(action_offset(m_actionSize, m_actionCount))
  {
    RFASSERT(0 < m_actionCount);
    RFASSERT(0 < m_stateSize);
    RFASSERT(actions.size() == feature_size(m_actionSize, m_actionCount));
  }

  DataType::const_iterator operator[](std::uint32_t action_index) const{
    return m_actions.begin() + m_actionOffset[action_index];
  }

  const DataType& state() const{
    return m_state;
  }

  DataType::const_iterator worst_action() const{
    return (*this)[m_actionCount - 1];
  } 

  double q_value(std::uint32_t action_index = 0) const{
    RFASSERT(action_index < m_actions.size());
    return m_actions[m_qValueOffset[action_index]];
  }

  double avg_q_value() const{
    double count = 0.0;
    double sum = 0.0;
    for(std::uint32_t action_index = 0; action_index < m_actionCount; ++action_index){
      if(
        (m_actionSize > std::count((*this)[action_index], (*this)[action_index] + m_actionSize, 0.0))
        &&(0 < q_value(action_index))
      ){
        sum += q_value(action_index);
        count += 1.0;
      }
    }
    if(0 < count)
      return sum / static_cast<double>(m_actionCount);
      else return sum;
  }

  double max_q_value() const{
    double max = q_value(0);
    for(std::uint32_t action_index = 1; action_index < m_actionCount; ++action_index){
      max = std::max(max, q_value(action_index));
    }
    return max;
  }

  double min_q_value() const{
    double min = q_value(0);
    for(std::uint32_t action_index = 1; action_index < m_actionCount; ++action_index){
      min = std::min(min, q_value(action_index));
    }
    return min;
  }

  std::uint32_t action_count(){
    return m_actionCount;
  }

  static constexpr std::uint32_t action_slot_size(std::uint32_t action_size){
    return (action_size + 1);
  }

  static constexpr std::uint32_t feature_size(std::uint32_t action_size, std::uint32_t action_count){
    return action_slot_size(action_size) * action_count;
  }

  static DataType action_slot(const DataType& action, double q_value){
    DataType ret(action);
    ret.insert(ret.begin(), q_value);
    return ret;
  }

protected:
  const std::uint32_t m_actionCount;
  const DataType& m_state;
  const DataType& m_actions;
  const std::uint32_t m_stateSize;
  const std::uint32_t m_actionSize;
  const std::vector<std::uint32_t> m_qValueOffset;
  const std::vector<std::uint32_t> m_actionOffset;
};

/**
 * @brief      This class helps handle a state paired with a number of Action Q-value pairs
 */
class RAFKO_EXPORT RafQSetItemView : public RafQSetItemConstView
{
  using RafQSetItemConstView::m_stateSize;
  using RafQSetItemConstView::m_actionSize;
  using RafQSetItemConstView::m_qValueOffset;
  using RafQSetItemConstView::m_actionOffset;
public:
  using DataType = RafQEnvironment::DataType;
  using RafQSetItemConstView::state;
  using RafQSetItemConstView::worst_action;
  using RafQSetItemConstView::q_value;
  using RafQSetItemConstView::avg_q_value;
  using RafQSetItemConstView::max_q_value;
  using RafQSetItemConstView::min_q_value;
  using RafQSetItemConstView::feature_size;
  using RafQSetItemConstView::action_slot;

  RafQSetItemView(
    DataType& state, DataType& actions,
    std::uint32_t action_size, std::uint32_t action_count = 1
  )
  : RafQSetItemConstView(state, actions, action_size, action_count)
  , m_state(state)
  , m_actions(actions)
  {
    RFASSERT(0 < m_actionCount);
  }

  void set_q_value(double value, std::uint32_t action_index = 0){
    RFASSERT(action_index < m_actions.size());
    m_actions[m_qValueOffset[action_index]] = value;
  }

  typename DataType::iterator operator[](std::uint32_t action_index){
    return m_actions.begin() + m_actionOffset[action_index];
  }

  typename DataType::iterator worst_action(){
    return (*this)[m_actionCount - 1];
  }

  typename DataType::iterator best_action(){
    return (*this)[m_actionCount - 1];
  }

  void copy_action(std::uint32_t source, std::uint32_t target){
    RFASSERT(source < m_actionCount);
    RFASSERT(target < m_actionCount);
    if(source == target) return;
    std::copy((*this)[source], (*this)[source] + m_actionSize, (*this)[target]);
    set_q_value(q_value(source), target);
  }

  void swap_action(std::uint32_t source, std::uint32_t target){
    RFASSERT(source < m_actionCount);
    RFASSERT(target < m_actionCount);
    if(source == target) return;
    std::swap_ranges((*this)[source], (*this)[source] + m_actionSize, (*this)[target]);
    double source_q_value = q_value(source);
    set_q_value(q_value(target), source);
    set_q_value(source_q_value, target);
  }


  void take_over(RafQSetItemConstView xp_element, std::uint32_t source_action_index = 0u, std::uint32_t target_action_index = 0u){
    RFASSERT(source_action_index < xp_element.action_count());
    RFASSERT(target_action_index < m_actionCount);
    std::copy(
      xp_element[source_action_index], xp_element[source_action_index] + m_actionSize, 
      (*this)[target_action_index]
    );
  }

  static typename DataType::iterator action_iterator(DataType& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index) + 1);
  }

  static typename DataType::iterator q_value_iterator(DataType& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index));
  }

private: 
  DataType& m_state;
  DataType& m_actions;
};

//TODO: convert state-best action pairs to sequences: so that consecutive states can be a sequence
//TODO: implement look_up function in openCL also

/**
 * @brief      This class stores and provides a set of states and connected actions with corresponding QValues
 */
class RAFKO_EXPORT RafQSet : public RafkoDataSet
{
public:
  using DataType = RafQEnvironment::DataType;
  using DataView = RafQEnvironment::DataView;
  using MaybeDataType = std::optional<std::reference_wrapper<const DataType>>; 

  RafQSet(
    const rafko_mainframe::RafkoSettings& settings, RafQEnvironment& environment,
    std::uint32_t action_count, std::uint32_t max_set_size, double overwrite_q_threshold
  )
  : m_settings(settings)
  , m_actionCount(action_count)
  , m_environment(environment)
  , m_costFunction(m_settings)
  , m_overwriteQThreshold(overwrite_q_threshold)
  , m_maxSetSize(max_set_size)
  {
    RFASSERT(0 < m_actionCount);
  }

  RafQSet(const RafQSet& other, std::uint32_t action_count)
  : m_settings(other.m_settings)
  , m_actionCount(action_count)
  , m_environment(other.m_environment)
  , m_costFunction(m_settings)
  , m_overwriteQThreshold(other.m_overwriteQThreshold)
  , m_maxSetSize(other.m_maxSetSize)
  {
    RFASSERT(m_actionCount <= action_count);
    for(std::uint32_t item_index = 0; item_index < other.get_number_of_sequences(); ++item_index){
      m_stateBuffer.push_back(other.m_stateBuffer[item_index]);
      m_actionsBuffer.push_back({
        other.m_actionsBuffer[item_index].begin(), 
        other.m_actionsBuffer[item_index].begin() + (m_actionCount * get_feature_size())
      });
      m_avgQValue.push_back(other.m_avgQValue[item_index]);
    }
  }

  MaybeDataType look_up(DataView state, std::uint32_t* result_index_buffer = nullptr) const{
    RFASSERT(state.size() == m_environment.state_size());
    //TODO: multi-thread below logic
    MaybeDataType result;
    for(std::uint32_t match_index = 0; match_index < get_number_of_sequences(); ++match_index){
      if(m_costFunction.get_feature_error(state, get_input_sample(match_index), m_environment.state_size()) <= m_settings.get_delta()){
        result.emplace(get_input_sample(match_index));
        if(result_index_buffer)
          *result_index_buffer = match_index;
        break; /* if the difference is small enough, a match is found! */
      }
    }
    return result;
  }

  void incorporate(const std::vector<DataType>& state_buffer, const std::vector<DataType>& actions_buffer){
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
    //TODO: multi-thread below logic
    for(std::uint32_t state_index = 0; state_index < state_buffer.size(); ++state_index){
      // std::cout << "Incorporating vector of " << state_buffer.size() << std::endl;
      RFASSERT(state_buffer[state_index].size() == get_input_size());
      RFASSERT(actions_buffer[state_index].size() == RafQSetItemView::feature_size(m_environment.action_size(), 1u/*action count*/));
      std::uint32_t match_index;
      MaybeDataType state_match = look_up(state_buffer[state_index], &match_index);
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
      }else{ /* no match is found for the state, extend the database with the newly found state*/
        m_stateBuffer.emplace_back(state_buffer[state_index]);
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

  void keep_best(std::uint32_t count){
    if(count < get_number_of_sequences())
      erase_worst(get_number_of_sequences() - count);
  }

  void erase_worst(std::uint32_t count){
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
      m_stateBuffer.erase(m_stateBuffer.begin() + index);
      m_actionsBuffer.erase(m_actionsBuffer.begin() + index);
    }
  }

  ~RafQSet() = default;

  RafQSetItemConstView operator[](std::uint32_t index) const{
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemConstView(
      m_stateBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  RafQSetItemView operator[](std::uint32_t index){
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemView(
      m_stateBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  constexpr std::uint32_t action_count(){
    return m_actionCount;
  }

  constexpr std::uint32_t max_size() const{
    return m_maxSetSize;
  }

  const DataType& get_input_sample(std::uint32_t raw_input_index) const override{
    RFASSERT( raw_input_index < get_number_of_sequences() );
    return m_stateBuffer[raw_input_index];
  }

  const std::vector<DataType>& get_input_samples() const override{
    return m_stateBuffer;
  }

  const DataType& get_label_sample(std::uint32_t raw_label_index) const override{
    RFASSERT( raw_label_index < get_number_of_sequences() );
    return m_actionsBuffer[raw_label_index];    
  }

  const std::vector<DataType>& get_label_samples() const override{
    return m_actionsBuffer;    
  }
  
  std::uint32_t get_input_size() const override{
    return m_environment.state_size();
  }

  std::uint32_t get_feature_size() const override{
    return RafQSetItemView::feature_size(m_environment.action_size(), m_actionCount);
  }

  std::uint32_t get_number_of_input_samples() const override{
    return m_stateBuffer.size();
  }

  std::uint32_t get_number_of_label_samples() const override{
    return m_actionsBuffer.size();
  }

  std::uint32_t get_number_of_sequences() const override{
    return m_stateBuffer.size();
  }
  
  std::uint32_t get_sequence_size() const override{
    return 1u;
  }

  std::uint32_t get_prefill_inputs_number() const override{
    return 0u;
  }

private:
  const rafko_mainframe::RafkoSettings& m_settings;
  const std::uint32_t m_actionCount;
  RafQEnvironment& m_environment;
  std::vector<DataType> m_stateBuffer;
  std::vector<DataType> m_actionsBuffer;
  std::vector<double> m_avgQValue;
  CostFunctionMSE m_costFunction;
  double m_overwriteQThreshold;
  std::uint32_t m_maxSetSize;

  //TODO: Experiment: each look ahead adds the delta ( improvement) of q-values
  double get_td_value(const RafQSetItemConstView& new_action_view, double old_q_value) const{
    // std::cout << "\n\n calculating td value.." << std::endl;
    double temporal_difference_value = new_action_view.q_value(0); /* Reqward: the only Q-value in @new_action_view */
    if(0 < m_settings.get_look_ahead_count()){
      double lambda = m_settings.get_gamma();
      std::uint32_t next_state_index;
      MaybeDataType next_state_data(new_action_view.state());
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
    //TODO: Learning rate may have an iteration, to interpolate
    return (temporal_difference_value - old_q_value) * m_settings.get_learning_rate();
  }
};

} /* namespace rafko_gym */
#endif /* RAFQ_SET_H */
