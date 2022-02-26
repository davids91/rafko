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
#if(RAFKO_USES_OPENCL)
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_environment.h"

#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_mainframe{

/**
 * @brief      Empty objective class
 */
class RafkoDummyObjective : public rafko_gym::RafkoObjective{
public:
  ~RafkoDummyObjective() = default;
  double set_feature_for_label(
    const rafko_gym::RafkoEnvironment& dataset, std::uint32_t sample_index,
    const std::vector<double>& neuron_data
  ) const{
    parameter_not_used(dataset);
    parameter_not_used(sample_index);
    parameter_not_used(neuron_data);
    return (0.0);
  }
  double set_features_for_labels(
     const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t raw_start_index, std::uint32_t labels_to_evaluate
  ) const{
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(raw_start_index);
    parameter_not_used(labels_to_evaluate);
    return (0.0);
  }
  double set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
    std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation
  ) const{
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(sequence_start_index);
    parameter_not_used(sequences_to_evaluate);
    parameter_not_used(start_index_in_sequence);
    parameter_not_used(sequence_truncation);
    return (0.0);
  }
  double set_features_for_sequences(
    const rafko_gym::RafkoEnvironment& dataset, const std::vector<std::vector<double>>& neuron_data,
    std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
    std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation,
    std::vector<double>& tmp_data
  ) const{
    parameter_not_used(dataset);
    parameter_not_used(neuron_data);
    parameter_not_used(neuron_buffer_index);
    parameter_not_used(sequence_start_index);
    parameter_not_used(sequences_to_evaluate);
    parameter_not_used(start_index_in_sequence);
    parameter_not_used(sequence_truncation);
    parameter_not_used(tmp_data);
    return (0.0);
  }

  #if(RAFKO_USES_OPENCL)
  void set_gpu_parameters(std::uint32_t pairs_to_evaluate_, std::uint32_t feature_size_){
    pairs_to_evaluate = pairs_to_evaluate_;
    feature_size = feature_size_;
  }
  cl::Program::Sources get_step_sources() const{
    return{R"(
      void kernel dummy_objective(
        __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
        __global double* outputs, __constant int* output_sizes, int output_sizes_size
      ){ }
    )"};
  }
  std::vector<std::string> get_step_names() const{
    return {"dummy_objective"};
  }
  std::vector<RafkoNBufShape> get_input_shapes() const{
    return { rafko_mainframe::RafkoNBufShape{ /* inputs and labels */
      pairs_to_evaluate * feature_size,
      pairs_to_evaluate * feature_size
    } };
  }
  std::vector<RafkoNBufShape> get_output_shapes() const{
    return {RafkoNBufShape{1}};
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const{
    return std::make_tuple(cl::NullRange,cl::NullRange,cl::NullRange);
  }
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  std::vector<double> dummy;
  std::uint32_t pairs_to_evaluate = 1u;
  std::uint32_t feature_size = 1u;
};

/**
 * @brief      Empty environment class
 */
 class RafkoDummyEnvironment : public rafko_gym::RafkoEnvironment{
 public:
   RafkoDummyEnvironment(std::uint32_t input_size_ = 1u, std::uint32_t feature_size_ = 1u)
   : dummy_inputs(1, std::vector<double>(input_size_))
   , dummy_labels(1, std::vector<double>(feature_size_))
   { }

   void push_state() { }
   void pop_state() { }
   const std::vector<double>& get_input_sample(std::uint32_t raw_input_index)const {
     parameter_not_used(raw_input_index);
     return dummy_inputs[0];
   }
   const std::vector<std::vector<double>>& get_input_samples()const { return dummy_inputs; }
   const std::vector<double>& get_label_sample(std::uint32_t raw_label_index)const {
     parameter_not_used(raw_label_index);
     return dummy_labels[0];
   }
   const std::vector<std::vector<double>>& get_label_samples()const { return dummy_labels; }
   std::uint32_t get_feature_size()const { return dummy_labels[0].size(); }
   std::uint32_t get_input_size()const { return dummy_labels[0].size(); }
   std::uint32_t get_number_of_input_samples()const { return 1; }
   std::uint32_t get_number_of_label_samples()const { return 1; }
   std::uint32_t get_number_of_sequences()const { return 1; }
   std::uint32_t get_sequence_size()const { return 1; }
   std::uint32_t get_prefill_inputs_number()const { return 0; }
   ~RafkoDummyEnvironment() = default;
 private:
   std::vector<std::vector<double>> dummy_inputs;
   std::vector<std::vector<double>> dummy_labels;
 };

#if(RAFKO_USES_OPENCL)
class RafkoDummyGPUStrategyPhase : public RafkoGPUStrategyPhase{
public:
  RafkoDummyGPUStrategyPhase(RafkoNBufShape input_shape_, RafkoNBufShape output_shape_)
  : input_shape(input_shape_)
  , output_shape(output_shape_)
  { }

  cl::Program::Sources get_step_sources() const{
    return{R"(
      void kernel dummy_kernel(
        __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
        __global double* outputs, __constant int* output_sizes, int output_sizes_size
      ){ }
    )"};
  }
  std::vector<std::string> get_step_names() const{
    return {"dummy_kernel"};
  }
  std::vector<RafkoNBufShape> get_input_shapes() const{
    return {input_shape};
  }
  std::vector<RafkoNBufShape> get_output_shapes() const{
    return {output_shape};
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const{
    return std::make_tuple(cl::NullRange,cl::NullRange,cl::NullRange);
  }
  ~RafkoDummyGPUStrategyPhase() = default;
private:
  const RafkoNBufShape input_shape;
  const RafkoNBufShape output_shape;
};
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_mainframe */

#endif /* RAFKO_DUMMIES_H */
