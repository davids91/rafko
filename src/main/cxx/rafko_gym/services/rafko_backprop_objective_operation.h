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

#ifndef RAFKO_BACKPROP_OBJECTIVE_OPERATION_H
#define RAFKO_BACKPROP_OBJECTIVE_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropObjectiveOperation{
public:
  RafkoBackpropObjectiveOperation(
    RafkoBackPropagation& parent, const rafko_net::RafkoNet& network,
    RafkoObjective& objective, std::uint32_t operation_index,
    std::uint32_t label_index_,  std::uint32_t sample_number_,
  ):RafkoBackpropagationOperation(parent, network, 0u, operation_index) /* Objective is always for the present value */
  , label_index(label_index_)
  , sample_number(sample_number_)
  {
  }

  void upload_dependencies_to_operations(){
    feature_dependency = parent.push_dependency(
      ad_operation_neuron_spike_d,
      (network.neuron_array_size() - network.output_neuron_number() + label_index)
    );
    set_registered();
  }

  void calculate(
    std::uint32 d_w_index, std::uint32 run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    RFASSERT(label_index < label_value[run_index].size());
    set_derivative(run_index, operation_index, d_w_index, objective.get_derivative(
      label_value[run_index][label_index], feature_dependency->get_value(run_index),
      feature_dependency->get_derivative(run_index, operation_index, d_w_index),
      static_cast<double>(sample_number)
    ));
    set_processed();
  }

private:
  const std::uint32_t label_index;
  const std::uint32_t sample_number;
  std::shared_ptr<RafkoBackpropagationOperation> feature_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_OBJECTIVE_OPERATION_H */
