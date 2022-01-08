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

#ifndef RAFKO_DUMMIES_H
#define RAFKO_DUMMIES_H

#include "rafko_global.h"

#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_environment.h"

namespace rafko_mainframe{

/**
 * @brief      Empty objective class
 */
class RafkoDummyObjective : public rafko_gym::RafkoObjective{
public:
  ~RafkoDummyObjective() = default;
  sdouble32 set_feature_for_label(const rafko_gym::RafkoEnvironment& dataset, uint32 sample_index, const std::vector<sdouble32>& neuron_data){
    parameter_not_used(dataset);
    parameter_not_used(sample_index);
    parameter_not_used(neuron_data);
    return double_literal(0.0);
  }
  sdouble32 set_features_for_labels(
     const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 raw_start_index, uint32 labels_to_evaluate
  ){
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(raw_start_index);
    parameter_not_used(labels_to_evaluate);
    return double_literal(0.0);
  }
  sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation
  ){
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(sequence_start_index);
    parameter_not_used(sequences_to_evaluate);
    parameter_not_used(start_index_in_sequence);
    parameter_not_used(sequence_truncation);
    return double_literal(0.0);
  }
  sdouble32 set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<sdouble32>>& neuron_data,
    uint32 neuron_buffer_index, uint32 sequence_start_index, uint32 sequences_to_evaluate,
    uint32 start_index_in_sequence, uint32 sequence_truncation,
    std::vector<sdouble32>& tmp_data
  ){
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(sequence_start_index);
    parameter_not_used(sequences_to_evaluate);
    parameter_not_used(start_index_in_sequence);
    parameter_not_used(sequence_truncation);
    parameter_not_used(tmp_data);
    return double_literal(0.0);
  }
private:
  std::vector<sdouble32> dummy;
};

/**
 * @brief      Empty environment class
 */
 class RafkoDummyEnvironment : public rafko_gym::RafkoEnvironment{
 public:
   RafkoDummyEnvironment(uint32 input_size_ = 1u, uint32 feature_size_ = 1u)
   : dummy_inputs(1, std::vector<sdouble32>(input_size_))
   , dummy_labels(1, std::vector<sdouble32>(feature_size_))
   { }

   void push_state() { }
   void pop_state() { }
   const std::vector<sdouble32>& get_input_sample(uint32 raw_input_index)const {
     parameter_not_used(raw_input_index);
     return dummy_inputs[0];
   }
   const std::vector<std::vector<sdouble32>>& get_input_samples()const { return dummy_inputs; }
   const std::vector<sdouble32>& get_label_sample(uint32 raw_label_index)const {
     parameter_not_used(raw_label_index);
     return dummy_labels[0];
   }
   const std::vector<std::vector<sdouble32>>& get_label_samples()const { return dummy_labels; }
   uint32 get_feature_size()const { return dummy_labels[0].size(); }
   uint32 get_input_size()const { return dummy_labels[0].size(); }
   uint32 get_number_of_input_samples()const { return 1; }
   uint32 get_number_of_label_samples()const { return 1; }
   uint32 get_number_of_sequences()const { return 1; }
   uint32 get_sequence_size()const { return 1; }
   uint32 get_prefill_inputs_number()const { return 0; }
   ~RafkoDummyEnvironment() = default;
 private:
   std::vector<std::vector<sdouble32>> dummy_inputs{{0}};
   std::vector<std::vector<sdouble32>> dummy_labels{{0}};
 };
} /* namespace rafko_mainframe */

#endif /* RAFKO_DUMMIES_H */
