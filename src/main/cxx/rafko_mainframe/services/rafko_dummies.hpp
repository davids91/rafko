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

#include "rafko_global.hpp"
#if(RAFKO_USES_OPENCL)
#include <CL/opencl.hpp>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/models/rafko_environment.hpp"

#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.hpp"
#endif/*(RAFKO_USES_OPENCL)*/

namespace rafko_mainframe {

/**
 * @brief      Empty environment class
 */
class RafkoDummyEnvironment : public rafko_gym::RafkoEnvironment{
  public:
    RafkoDummyEnvironment(std::uint32_t input_size = 1u, std::uint32_t feature_size = 1u)
    : m_dummyInputs(1, std::vector<double>(input_size))
    , m_dummyLabels(1, std::vector<double>(feature_size))
    { }

    void push_state() override{ }
    void pop_state() override{ }
    const std::vector<double>& get_input_sample(std::uint32_t /*raw_input_index*/)const override{
      return m_dummyInputs[0];
    }
    const std::vector<std::vector<double>>& get_input_samples()const override{ return m_dummyInputs; }
    const std::vector<double>& get_label_sample(std::uint32_t /*raw_label_index*/)const override{
      return m_dummyLabels[0];
    }
    const std::vector<std::vector<double>>& get_label_samples()const override{ return m_dummyLabels; }
    std::uint32_t get_feature_size()const override{ return m_dummyLabels[0].size(); }
    std::uint32_t get_input_size()const override{ return m_dummyLabels[0].size(); }
    std::uint32_t get_number_of_input_samples()const override{ return 1; }
    std::uint32_t get_number_of_label_samples()const override{ return 1; }
    std::uint32_t get_number_of_sequences()const override{ return 1; }
    std::uint32_t get_sequence_size()const override{ return 1; }
    std::uint32_t get_prefill_inputs_number()const override{ return 0; }
    ~RafkoDummyEnvironment() = default;
  private:
    std::vector<std::vector<double>> m_dummyInputs;
    std::vector<std::vector<double>> m_dummyLabels;
};

#if(RAFKO_USES_OPENCL)
class RafkoDummyGPUStrategyPhase : public RafkoGPUStrategyPhase{
public:
  RafkoDummyGPUStrategyPhase(RafkoNBufShape input_shape, RafkoNBufShape output_shape)
  : m_inputShape(input_shape)
  , m_outputShape(output_shape)
  { }

  cl::Program::Sources get_step_sources() const override{
    return{R"(
      void kernel dummy_kernel(
        __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
        __global double* outputs, __constant int* output_sizes, int output_sizes_size
      ){ }
    )"};
  }
  std::vector<std::string> get_step_names() const override{
    return {"dummy_kernel"};
  }
  std::vector<RafkoNBufShape> get_input_shapes() const override{
    return {m_inputShape};
  }
  std::vector<RafkoNBufShape> get_output_shapes() const override{
    return {m_outputShape};
  }
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const override{
    return std::make_tuple(cl::NullRange,cl::NullRange,cl::NullRange);
  }
  ~RafkoDummyGPUStrategyPhase() = default;
private:
  const RafkoNBufShape m_inputShape;
  const RafkoNBufShape m_outputShape;
};
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_mainframe */

#endif /* RAFKO_DUMMIES_H */
