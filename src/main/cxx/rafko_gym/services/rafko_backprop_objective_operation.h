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

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropObjectiveOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropObjectiveOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    RafkoObjective& objective_, std::uint32_t operation_index,
    std::uint32_t output_index_,  std::uint32_t sample_number_
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , objective(objective_)
  , output_index(output_index_)
  , sample_number(sample_number_)
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    return {{
      {{ad_operation_neuron_spike_d, {(network.neuron_array_size() - network.output_neuron_number() + output_index)}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        feature_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ){
    parameter_not_used(network_input);
    RFASSERT(output_index < label_data.size());
    RFASSERT(static_cast<bool>(feature_dependency));
    set_derivative(d_w_index, objective.get_derivative(
      label_data[output_index], feature_dependency->get_value(0u/*past_index*/),
      feature_dependency->get_derivative(0u/*past_index*/, d_w_index), static_cast<double>(sample_number)
    ));
    //TODO: Do we need to have values for this?
    set_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    return "";
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {feature_dependency};
  }

private:
  RafkoObjective& objective;
  const std::uint32_t output_index;
  const std::uint32_t sample_number;
  std::shared_ptr<RafkoBackpropagationOperation> feature_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_OBJECTIVE_OPERATION_H */
