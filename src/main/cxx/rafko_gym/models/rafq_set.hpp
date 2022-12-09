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
  using FeatureVector = RafQEnvironment::FeatureVector;

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

  FeatureVector::const_iterator operator[](std::uint32_t action_index) const{
    return m_actions.begin() + m_actionOffset[action_index];
  }

  const FeatureVector& state() const{
    return m_state;
  }

  FeatureVector::const_iterator worst_action() const{
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

  static FeatureVector action_slot(const FeatureVector& action, double q_value){
    FeatureVector ret(action);
    ret.insert(ret.begin(), q_value);
    return ret;
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
 * @brief      This class helps handle a state paired with a number of Action Q-value pairs
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

  void set_q_value(double value, std::uint32_t action_index = 0){
    RFASSERT(action_index < m_actions.size());
    m_actions[m_qValueOffset[action_index]] = value;
  }

  typename FeatureVector::iterator operator[](std::uint32_t action_index){
    return m_actions.begin() + m_actionOffset[action_index];
  }

  typename FeatureVector::iterator worst_action(){
    return (*this)[m_actionCount - 1];
  }

  typename FeatureVector::iterator best_action(){
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

  static typename FeatureVector::iterator action_iterator(FeatureVector& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index) + 1);
  }

  static typename FeatureVector::iterator q_value_iterator(FeatureVector& actions_buffer, std::uint32_t action_size, std::uint32_t action_index = 0){ 
    RFASSERT((((action_size + 1) * action_index) + action_size + 1) <= actions_buffer.size());
              /* Action slot start + (slot size *  slot_index )         + q_value offset */
    return (actions_buffer.begin() + ((action_size + 1) * action_index));
  }

private: 
  FeatureVector& m_state;
  FeatureVector& m_actions;
};

//TODO: convert state-best action pairs to sequences: so that consecutive states can be a sequence
//TODO: generate QSet from @Dataset ( make save/load of qSets possible )

/**
 * @brief      This class stores and provides a set of states and connected actions with corresponding QValues
 */
class RAFKO_EXPORT RafQSet : public RafkoDataSet
{
public:
  using FeatureVector = RafQEnvironment::FeatureVector;
  using FeatureView = RafQEnvironment::FeatureView;
  using MaybeFeatureVector = std::optional<std::reference_wrapper<const FeatureVector>>; 

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
  , m_lookupThreads(m_settings.get_max_solve_threads()) /* because cost function uses get_max_solve_threads */
  {
    RFASSERT(0 < m_actionCount);
    m_statesBuffer.reserve(m_maxSetSize);
    m_actionsBuffer.reserve(m_maxSetSize);
  }

  RafQSet(const RafQSet& other, std::uint32_t action_count)
  : m_settings(other.m_settings)
  , m_actionCount(action_count)
  , m_environment(other.m_environment)
  , m_costFunction(m_settings)
  , m_overwriteQThreshold(other.m_overwriteQThreshold)
  , m_maxSetSize(other.m_maxSetSize)
  , m_lookupThreads(m_settings.get_max_solve_threads())
  {
    RFASSERT(m_actionCount <= action_count);
    m_statesBuffer.reserve(m_maxSetSize);
    m_actionsBuffer.reserve(m_maxSetSize);
    for(std::uint32_t item_index = 0; item_index < other.get_number_of_sequences(); ++item_index){
      m_statesBuffer.push_back(other.m_statesBuffer[item_index]);
      m_actionsBuffer.push_back({
        other.m_actionsBuffer[item_index].begin(), 
        other.m_actionsBuffer[item_index].begin() + (m_actionCount * get_feature_size())
      });
      m_avgQValue.push_back(other.m_avgQValue[item_index]);
    }
  }

  MaybeFeatureVector look_up(FeatureView state, std::uint32_t* result_index_buffer = nullptr) const;

  //TODO: progress callback
  void incorporate(
    const std::vector<FeatureVector>& state_buffer, const std::vector<FeatureVector>& actions_buffer, 
    const std::function<void(double/*progress*/)>& progress_callback = {}
  );

  void keep_best(std::uint32_t count){
    if(count < get_number_of_sequences())
      erase_worst(get_number_of_sequences() - count);
  }

  void erase_worst(std::uint32_t count);

  ~RafQSet() = default;

  RafQSetItemConstView operator[](std::uint32_t index) const{
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemConstView(
      m_statesBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  RafQSetItemView operator[](std::uint32_t index){
    RFASSERT(index < get_number_of_sequences());
    return RafQSetItemView(
      m_statesBuffer[index], m_actionsBuffer[index], 
      m_environment.action_size(), m_actionCount
    );
  }

  constexpr std::uint32_t action_count(){
    return m_actionCount;
  }

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
  std::vector<double> m_avgQValue;
  CostFunctionMSE m_costFunction;
  double m_overwriteQThreshold;
  std::uint32_t m_maxSetSize;
  rafko_utilities::ThreadGroup m_lookupThreads; 

  double get_td_value(const RafQSetItemConstView& new_action_view, double old_q_value) const;
};

} /* namespace rafko_gym */
#endif /* RAFQ_SET_H */
