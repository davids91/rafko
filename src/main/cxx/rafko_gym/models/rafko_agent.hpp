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

#ifndef RAFKO_AGENT_H
#define RAFKO_AGENT_H

#include "rafko_global.hpp"

#include <functional>
#include <vector>
#if (RAFKO_USES_OPENCL)
#include <CL/opencl.hpp>
#include <numeric>
#endif /*(RAFKO_USES_OPENCL)*/

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#include "rafko_utilities/models/data_pool.hpp"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#if (RAFKO_USES_OPENCL)
#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"
#include "rafko_mainframe/models/rafko_nbuf_shape.hpp"
#include "rafko_net/services/solution_builder.hpp"
#endif /*(RAFKO_USES_OPENCL)*/

namespace rafko_gym {

/**
 * @brief      This class serves as a base for reinforcement learning agent,
 * which provides output data based on different inputs
 */
class RAFKO_EXPORT RafkoAgent
#if (RAFKO_USES_OPENCL)
    : public rafko_mainframe::RafkoGPUStrategy
#endif /*(RAFKO_USES_OPENCL)*/
{
public:
  RafkoAgent(const rafko_mainframe::RafkoSettings &settings)
      : m_settings(settings) {}

  virtual ~RafkoAgent() = default;

  /*
   * @brief   Sets evaluation mode for agent, which signals whether or not
   * training relevant Neural features(e.g. dropout) are to be executed.
   *
   * @param[in]   evaluation    decides whether the agent is in evaluation mode
   * or not
   *
   */
  virtual void set_eval_mode(bool evaluation) = 0;

  /**
   * @brief      For the provided input, return the result of the neural network
   *
   * @param[in]      input                  The input data to be taken
   * @param[in]      reset_neuron_data      should the internal memory of the
   * solver is to be resetted before solving the neural network
   * @param[in]      thread_index           The index of thread the solution is
   * to be running from
   *
   * @return         The output values of the network result
   */
  virtual rafko_utilities::ConstVectorSubrange<>
  solve(const std::vector<double> &input, bool reset_neuron_data = false,
        std::uint32_t thread_index = 0u) = 0;

#if (RAFKO_USES_OPENCL) /* Methods forwarding from                             \
                           rafko_mainframe::RafkoGPUStrategy */
  virtual cl::Program::Sources get_step_sources() const override = 0;
  virtual std::vector<std::string> get_step_names() const override = 0;
  virtual std::vector<rafko_mainframe::RafkoNBufShape>
  get_input_shapes() const override = 0;
  virtual std::vector<rafko_mainframe::RafkoNBufShape>
  get_output_shapes() const override = 0;
  virtual std::tuple<cl::NDRange, cl::NDRange, cl::NDRange>
  get_solution_space() const override = 0;
#endif /*(RAFKO_USES_OPENCL)*/

protected:
  const rafko_mainframe::RafkoSettings &m_settings;
};

} /* namespace rafko_gym */
#endif /* RAFKO_AGENT_H */
