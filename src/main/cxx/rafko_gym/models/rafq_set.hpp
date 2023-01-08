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
#include <functional>
#include <optional>
#include <mutex>
#include <functional>

#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/models/rafko_dataset_implementation.hpp"
#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/services/cost_function_mse.hpp"

namespace rafko_gym{

/**
 * @brief      This class helps query a state paired with a number of Action Q-value pairs with only const access
 */
class RAFKO_EXPORT RafQSetItemConstView
{
  /**
   * @brief     Calculates the offsets for the q-values in the common buffer based on the given action size and count
   */
  static std::vector<std::uint32_t> q_value_offset(std::uint32_t  action_size, std::uint32_t action_count){
    std::vector<std::uint32_t> offsets;
    for(std::uint32_t action_index = 0u; action_index < action_count; ++action_index)
      offsets.push_back( action_index * (action_size + 1u) );
    return offsets;
  }

  /**
   * @brief     Calculates the offsets for the actions in the common buffer based on the given action size and count
   */
  static std::vector<std::uint32_t> action_offset(std::uint32_t  action_size, std::uint32_t action_count){
    std::vector<std::uint32_t> offsets;
    for(std::uint32_t action_index = 0u; action_index < action_count; ++action_index)
      offsets.push_back( action_index * (action_size + 1u) + 1u );
    return offsets;
  }

public:
  using FeatureVector = RafQEnvironment::FeatureVector;
  using FeatureView = RafQEnvironment::FeatureView;

  RafQSetItemConstView(
    const FeatureVector& state, const FeatureVector& actions,
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

  /**
   * @brief     Provides const access for actions
   * 
   * @param[in]     Action_index    the index of the action to query
   * 
   * @return    Const iterator to the first element of the action under the given index
   */
  FeatureVector::const_iterator operator[](std::uint32_t action_index) const{
    return m_actions.begin() + m_actionOffset[action_index];
  }

  /**
   * @brief     Provides const access for the enclosed state
   * 
   * @return    Const reference to the enclosed state vector
   */
  const FeatureVector& state() const{
    return m_state;
  }

  /**
   * @brief     Provides const access for actions
   * 
   * @param[in]     Action_index    the index of the action to query
   * 
   * @return    A view the buffer the queried action is stored in
   */
  FeatureView action(std::uint32_t action_index = 0u) const{
    RFASSERT(action_index < m_actionCount);
    return {(*this)[action_index], m_actionSize};
  }

  /**
   * @brief     Provides const access for actions
   * 
   * @return    Const iterator to the first element of the worst action
   */
  FeatureVector::const_iterator worst_action() const{
    return (*this)[m_actionCount - 1];
  } 

  /**
   * @brief     Provides access to the stored q-values
   * 
   * @param[in]     action_index    The index of the action to query(q-values are paired with actions)
   * 
   * @return    The value of the q-value stored under the given action index
   */
  double q_value(std::uint32_t action_index = 0) const{
    RFASSERT(action_index < m_actions.size());
    return m_actions[m_qValueOffset[action_index]];
  }

  /**
   * @brief     Calculates the average of the enclosed q-values
   * 
   * @return    The average value of the enclosed q-values
   */
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

  /**
   * @brief     Calculates the maximum of the enclosed q-values
   * 
   * @return    The maximum value of the enclosed q-values
   */
  double max_q_value() const{
    double max = q_value(0);
    for(std::uint32_t action_index = 1; action_index < m_actionCount; ++action_index){
      max = std::max(max, q_value(action_index));
    }
    return max;
  }

  /**
   * @brief     Calculates the minimum of the enclosed q-values
   * 
   * @return    The minimum value of the enclosed q-values
   */
  double min_q_value() const{
    double min = q_value(0);
    for(std::uint32_t action_index = 1; action_index < m_actionCount; ++action_index){
      min = std::min(min, q_value(action_index));
    }
    return min;
  }

  /**
   * @brief     Provides the number of actions paired to one state in this view
   * 
   * @return    Number of actions kept track
   */
  std::uint32_t action_count(){
    return m_actionCount;
  }

  /**
   * @brief     Provides the size of one action-q-value pair
   * 
   * @return    Number of elements in one action-q-value pair
   */
  static constexpr std::uint32_t action_slot_size(std::uint32_t action_size){
    return (action_size + 1);
  }

  /**
   * @brief     Provides the number of elements stored in the enclosed actions buffer
   * 
   * @return    The number of elements making up all of the enclosed actions and q-values
   */
  static constexpr std::uint32_t feature_size(std::uint32_t action_size, std::uint32_t action_count){
    return action_slot_size(action_size) * action_count;
  }

  /**
   * @brief     Generates a vector from the given data in a size-agnostic way
   * 
   * @param[in]     action      Const reference to the buffer containing the actions
   * @param         q_value     The q-value to assign the action buffer    
   */
  static FeatureVector action_slot(const FeatureVector& action, double q_value){
    FeatureVector ret(action.size() + 1);
    std::copy(action.begin(), action.end(), ret.begin() + 1);
    ret[0] = q_value;
    return ret;
  }

  /**
   * @brief     Gives back a view of the best action from given actions buffer. 
   *            Buffer is considered to contain at least one action of the given size
   * 
   * @param[in]     actions_buffer    The buffer to query
   * @param         action_size       Size of one Action inside the buffer
   * 
   * @return    A view of the best action inside the given buffer
   */
  static FeatureView best_action_slot(FeatureView actions_buffer, std::uint32_t action_size){
    RFASSERT(action_slot_size(action_size) <= actions_buffer.size());
    return{ actions_buffer.begin() + 1, actions_buffer.begin() + 1 + action_size };
  }

protected:
  const std::uint32_t m_actionCount;
  const FeatureVector& m_state;
  const FeatureVector& m_actions;
  const std::uint32_t m_stateSize;
  const std::uint32_t m_actionSize;
  const std::vector<std::uint32_t> m_qValueOffset;
  const std::vector<std::uint32_t> m_actionOffset;
};

/**
 * @brief      This class helps read and update a state paired with a number of Action Q-value pairs
 */
class RAFKO_EXPORT RafQSetItemView : public RafQSetItemConstView
{
  using RafQSetItemConstView::m_stateSize;
  using RafQSetItemConstView::m_actionSize;
  using RafQSetItemConstView::m_qValueOffset;
  using RafQSetItemConstView::m_actionOffset;
public:
  using FeatureVector = RafQEnvironment::FeatureVector;
  using RafQSetItemConstView::state;
  using RafQSetItemConstView::worst_action;
  using RafQSetItemConstView::q_value;
  using RafQSetItemConstView::avg_q_value;
  using RafQSetItemConstView::max_q_value;
  using RafQSetItemConstView::min_q_value;
  using RafQSetItemConstView::feature_size;
  using RafQSetItemConstView::action_slot;

  RafQSetItemView(
    FeatureVector& state, FeatureVector& actions,
    std::uint32_t action_size, std::uint32_t action_count = 1
  )
  : RafQSetItemConstView(state, actions, action_size, action_count)
  , m_state(state)
  , m_actions(actions)
  {
    RFASSERT(0 < m_actionCount);
  }

  /**
   * @brief     Updates the q-value under the given index
   * 
   * @param     value           The q-value to update the given action with
   * @param     action_index    The index of the action to update the q-value for
   */
  void set_q_value(double value, std::uint32_t action_index = 0){
    RFASSERT(action_index < m_actions.size());
    m_actions[m_qValueOffset[action_index]] = value;
  }

  /**
   * @brief     Provides access for actions
   * 
   * @param     action_index    The index of the action to rquery
   * 
   * @return    Iterator to the first element of the given action
   */
  typename FeatureVector::iterator operator[](std::uint32_t action_index){
    return m_actions.begin() + m_actionOffset[action_index];
  }

  /**
   * @brief     Provides access to the worst action
   * 
   * @return    Iterator to the first element of the worst action
   */
  typename FeatureVector::iterator worst_action(){
    return (*this)[m_actionCount - 1];
  }

  /**
   * @brief     Provides access to the best action
   * 
   * @return    Iterator to the first element of the best action
   */
  typename FeatureVector::iterator best_action(){
    return (*this)[m_actionCount - 1];
  }

  /**
   * @brief     Copies action buffer data from source to target
   * 
   * @param     source    The index of the action to copy the data from
   * @param     target    The index of the action to copy the data to
   */
  void copy_action(std::uint32_t source, std::uint32_t target){
    RFASSERT(source < m_actionCount);
    RFASSERT(target < m_actionCount);
    if(source == target) return;
    std::copy((*this)[source], (*this)[source] + m_actionSize, (*this)[target]);
    set_q_value(q_value(source), target);
  }

  /**
   * @brief     Swaps action buffer data from source to target
   * 
   * @param     source    The index of the action to swap
   * @param     target    The index of the action to swap it with
   */
  void swap_action(std::uint32_t source, std::uint32_t target){
    RFASSERT(source < m_actionCount);
    RFASSERT(target < m_actionCount);
    if(source == target) return;
    std::swap_ranges((*this)[source], (*this)[source] + m_actionSize, (*this)[target]);
    double source_q_value = q_value(source);
    set_q_value(q_value(target), source);
    set_q_value(source_q_value, target);
  }

  /**
   * @brief     Copies action buffer data from RafQSetItemConstView to target
   * 
   * @param     xp_element              The View to take over action data from
   * @param     source_action_index     The index of the action in @xp_element to copy the data from
   * @param     target_action_index     The index of the action in the current object to copy the data to
   */
  void take_over(RafQSetItemConstView xp_element, std::uint32_t source_action_index = 0u, std::uint32_t target_action_index = 0u){
    RFASSERT(source_action_index < xp_element.action_count());
    RFASSERT(target_action_index < m_actionCount);
    std::copy(
      xp_element[source_action_index], xp_element[source_action_index] + m_actionSize, 
      (*this)[target_action_index]
    );
  }

  /**
   * @brief     Provides access to the buffer of the action under the given action index. Indexes are checked for OOB
   * 
   * @param     actions_buffer      Rreference to the buffer containing the action-q-value pairs
   * @param     action_size         Size of one action inside the buffer
   * @param     action_index        The index of the action to reach
   * 
   * @return    Iterator to the first element of the queried action
   */
  static FeatureVector::iterator action_iterator(FeatureVector& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index) + 1);
  }

  /**
   * @brief     Provides access to the buffer of the q-value under the given action index. Indexes are checked for OOB
   * 
   * @param     actions_buffer      Rreference to the buffer containing the action-q-value pairs
   * @param     action_size         Size of one action inside the buffer
   * @param     action_index        The index of the action to reach
   * 
   * @return    Iterator to the first element of the queried action
   */
  static FeatureVector::iterator q_value_iterator(FeatureVector& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index));
  }

private: 
  FeatureVector& m_state;
  FeatureVector& m_actions;
};

/**
 * @brief      This class stores and provides a set of states and connected actions with corresponding QValues
 */
class RAFKO_EXPORT RafQSet : public RafkoDataSet
{
public:
  using FeatureVector = RafQEnvironment::FeatureVector;
  using FeatureView = RafQEnvironment::FeatureView;
  using MaybeFeatureVector = std::optional<std::reference_wrapper<const FeatureVector>>; 
  using AnyData = RafQEnvironment::AnyData;

  RafQSet(
    const rafko_mainframe::RafkoSettings& settings, RafQEnvironment& environment,
    std::uint32_t action_count, std::uint32_t max_set_size, double overwrite_q_threshold
  );

  RafQSet(const RafQSet& other, std::uint32_t action_count);

  RafQSet(
    const rafko_mainframe::RafkoSettings& settings, RafQEnvironment& environment, 
    std::uint32_t action_count, double overwrite_q_threshold, const DataSetPackage& source
  );

  /**
   * @brief     Export every item in the set into the @DataSetPackage message for later use
   * 
   * @return    A @DataSetPackage containing all of the stored items in the set with a sequence size of 1
   */
  DataSetPackage generate_package() const{
    return RafkoDatasetImplementation::generate_from(
      m_statesBuffer, m_actionsBuffer, get_sequence_size(), m_maxSetSize
    );
  }

  /**
   * @brief     Export the best action from the set into a @DataSetPackage for later use. 
   *            Produces an empty package if long enough sequences could not be generated
   * 
   * @param     preferred_sequence_size     The number of items expected to be in one sequence
   * 
   * @return    A @DataSetPackage containing all of the best actions in temporal order, or an empty package
   */
  DataSetPackage generate_best_sequences(std::uint32_t preferred_sequence_size) const;

  /**
   * @brief     Provides a View matching the given state View should there be a match inside the set.
   * 
   * @param[in]       state                   The data to look for in the stored states
   * @param[inout]    result_index_buffer     A pointer to an integer to store the result index should there be any
   * 
   * @return    An optional, storing the reference to the stored state buffer matching the given one, if there is any
   */
  MaybeFeatureVector look_up(FeatureView state, std::uint32_t* result_index_buffer = nullptr) const;

  /**
   * @brief     Updates the q-set with the given state-action-q-value pairs. The sizes of the given vectors 
   *            must match exactly.
   * 
   * @param[in]     state_buffer          A vector of states to look for
   * @param[in]     actions_buffer        A vector of action-q-value pairs to update the data with
   * @param[in]     user_data_buffer      Custom data for each state, to help environment restore state from hidden data
   * @param[in]     progress_callback     A function to call at each progress update with a value of 0..1 representing the progress of the operation
   */
  void incorporate(
    const std::vector<FeatureVector>& state_buffer, const std::vector<FeatureVector>& actions_buffer,
    std::vector<AnyData>&& user_data_buffer = {}, const std::function<void(double/*progress*/)>& progress_callback = [](double){}
  );

  /**
   * @brief     Erases elements from the set if it gets greater, than the given size. Elements with the smallest q-values are erased
   * 
   * @param     count     the number of elements to keep in the set
   */
  void keep_best(std::uint32_t count){
    if(count < get_number_of_sequences())
      erase_worst(get_number_of_sequences() - count);
  }

  /**
   * @brief     Erases the worst q-value elements from the set
   * 
   * @param     count     the number of elements to erase from the set
   */
  void erase_worst(std::uint32_t count);

  ~RafQSet() = default;

  /**
   * @brief     Provides const access to an item in the set under the given index
   * 
   * @param     index     The index of the item to query
   * 
   * @return  An instance of @RafQSetItemConstView for the elements under the given index
   */
  RafQSetItemConstView operator[](std::uint32_t index) const{
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemConstView(
      m_statesBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  /**
   * @brief     Provides access to an item in the set under the given index
   * 
   * @param     index     The index of the item to query
   * 
   * @return  An instance of @RafQSetItemView for the elements under the given index
   */
  RafQSetItemView operator[](std::uint32_t index){
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemView(
      m_statesBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  /**
   * @brief     Provides the number of action assigned for one state by the q-set policy
   * 
   * @return    Number of action-q-value pairs assigned to one state inside the q-set
   */
  constexpr std::uint32_t action_count(){
    return m_actionCount;
  }

  /**
   * @brief     Gives back the maximum number of elements the q-set is configured to contain
   * 
   * @return    Number of elements the set is configured to keep when extending
   */
  constexpr std::uint32_t max_size() const{
    return m_maxSetSize;
  }

  const FeatureVector& get_input_sample(std::uint32_t raw_input_index) const override{
    RFASSERT( raw_input_index < get_number_of_sequences() );
    return m_statesBuffer[raw_input_index];
  }

  const std::vector<FeatureVector>& get_input_samples() const override{
    return m_statesBuffer;
  }

  const FeatureVector& get_label_sample(std::uint32_t raw_label_index) const override{
    RFASSERT( raw_label_index < get_number_of_sequences() );
    return m_actionsBuffer[raw_label_index];    
  }

  const std::vector<FeatureVector>& get_label_samples() const override{
    return m_actionsBuffer;    
  }
  
  std::uint32_t get_input_size() const override{
    return m_environment.state_size();
  }

  std::uint32_t get_feature_size() const override{
    return RafQSetItemView::feature_size(m_environment.action_size(), m_actionCount);
  }

  std::uint32_t get_number_of_input_samples() const override{
    return m_statesBuffer.size();
  }

  std::uint32_t get_number_of_label_samples() const override{
    return m_actionsBuffer.size();
  }

  std::uint32_t get_number_of_sequences() const override{
    return m_statesBuffer.size();
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
  std::vector<FeatureVector> m_statesBuffer;
  std::vector<FeatureVector> m_actionsBuffer;
  std::vector<AnyData> m_userDataBuffer;
  std::vector<double> m_avgQValue;
  CostFunctionMSE m_costFunction;
  double m_overwriteQThreshold;
  std::uint32_t m_maxSetSize;
  rafko_utilities::ThreadGroup m_lookupThreads;
  mutable std::mutex m_searchResultMutex;

  /**
   * @brief     Calculates the Temporal difference value for the given state-action-q-value pair 
   * 
   * @param         new_action_view       Object providing const acess to the data for the given state-action-q-value pair
   * @param         old_q_value           The Q-value for the given object prior to a recent update
   * @param[in]     user_data_buffer      Custom data for the new action state, to help environment restore state from hidden data not included in the state vector
   */
  double get_td_value(const RafQSetItemConstView& new_action_view, double old_q_value, const AnyData& user_data = {}) const;
};

} /* namespace rafko_gym */
#endif /* RAFQ_SET_H */
