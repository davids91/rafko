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

#ifndef RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H
#define RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/models/transfer_function.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropTransferFnOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropTransferFnOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, rafko_mainframe::RafkoSettings& settings
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , transfer_function(settings)
  , neuron_index(neuron_index_)
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    return {{
      {{ad_operation_neuron_input_d, {neuron_index, 0u/*neuron_input_index*/}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        needed_input_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(needed_input_dependency));
    RFASSERT(needed_input_dependency->is_processed());
    set_value(run_index, transfer_function.get_value(
      network.neuron_array(neuron_index).transfer_function(), needed_input_dependency->get_value(run_index)
    ));
    set_derivative(run_index, d_w_index, transfer_function.get_derivative( /* d t(f(w))/dx = f'(w) * t'(f(w))*/
      network.neuron_array(neuron_index).transfer_function(),
      needed_input_dependency->get_value(run_index),
      needed_input_dependency->get_derivative(run_index, d_w_index)
    ));
    set_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    RFASSERT(static_cast<bool>(needed_input_dependency));
    RFASSERT(needed_input_dependency->are_dependencies_registered());
    return (
      "| \t transfer function[" + std::to_string(neuron_index) + "]: "
      + rafko_net::Transfer_functions_Name(network.neuron_array(neuron_index).transfer_function()) + "\n"
      + needed_input_dependency->value_kernel_function()
    );
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {needed_input_dependency};
  }

private:
  const rafko_net::TransferFunction transfer_function;
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> needed_input_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H */
