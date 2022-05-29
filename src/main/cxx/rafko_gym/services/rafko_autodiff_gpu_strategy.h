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

#ifndef RAFKO_AUTODIFF_GPU_STRATEGY_H
#define RAFKO_AUTODIFF_GPU_STRATEGY_H

#include "rafko_global.h"

#include <cmath>
#include <memory>
#include <string>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_nbuf_shape.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"

#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 */
class AutoDiffGPUStrategy
: public rafko_mainframe::RafkoGPUStrategyPhase
{
public:
  AutoDiffGPUStrategy(const rafko_mainframe::RafkoSettings& settings_, rafko_net::RafkoNet& network_)
  : settings(settings_)
  , network(network_)
  {
  }

  AutoDiffGPUStrategy(
    const rafko_mainframe::RafkoSettings& settings_, rafko_net::RafkoNet& network_,
    std::shared_ptr<RafkoEnvironment> environment_
  )
  : AutoDiffGPUStrategy(settings_, network_)
  {
    set_environment(environment_);
  }

  void set_environment(std::shared_ptr<RafkoEnvironment> environment_){
    environment = environment_;
    RFASSERT(environment->get_input_size() == network.input_data_size());
    used_minibatch_size = std::min(
      settings.get_minibatch_size(), environment->get_number_of_sequences()
    );
    built = false;
  }

  void build(std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations);

  cl::Program::Sources get_step_sources() const{
    RFASSERT(built);
    RFASSERT(static_cast<bool>(environment));
    return {built_source};
  }

  std::vector<std::string> get_step_names() const{
    return {"autodiff_iterate"};
  }

  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const{
    RFASSERT(static_cast<bool>(environment));
    return{ rafko_mainframe::RafkoNBufShape{
      /* Weights */
      static_cast<std::uint32_t>(network.weight_table_size()),
      /* Inputs */
      used_minibatch_size * environment->get_inputs_in_one_sequence() * network.input_data_size(),
      /* Labels */
      used_minibatch_size * environment->get_sequence_size() * network.output_neuron_number()
    } };
  }

  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const{
    /* Weight derivatives */
    return{ rafko_mainframe::RafkoNBufShape{static_cast<std::uint32_t>(network.weight_table_size())} };
  }

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> get_solution_space() const{
    RFASSERT(static_cast<bool>(environment));
    return {cl::NullRange/*offset*/, cl::NDRange(used_minibatch_size)/*global*/, cl::NullRange/*local*/ };
  }

private:
  const rafko_mainframe::RafkoSettings& settings;
  rafko_net::RafkoNet& network;
  std::shared_ptr<RafkoEnvironment> environment;
  std::uint32_t used_minibatch_size;
  bool built = false;
  std::string built_source;
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_GPU_STRATEGY_H */
