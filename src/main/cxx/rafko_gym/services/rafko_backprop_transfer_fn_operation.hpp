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

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/transfer_function.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.hpp"

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
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_, const rafko_mainframe::RafkoSettings& settings
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_neuron_transfer_d)
  , transfer_function(settings)
  , neuron_index(neuron_index_)
  {
  }
  ~RafkoBackpropTransferFnOperation() = default;

  DependencyRequest upload_dependencies_to_operations() override{
    return {{
      {{ad_operation_neuron_input_d, {neuron_index, 0u/*neuron_input_index*/}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        needed_input_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate_value(const std::vector<double>& network_input) override{
    parameter_not_used(network_input);
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(needed_input_dependency));
    RFASSERT(needed_input_dependency->is_value_processed());
    RFASSERT_LOG(
      "operation[{}]: Neuron[{}] Transfer function = {}({}(op[{}]))", get_operation_index(),
      neuron_index, Transfer_functions_Name(network.neuron_array(neuron_index).transfer_function()),
      needed_input_dependency->get_value(0u/*past_index*/), needed_input_dependency->get_operation_index()
    );
    set_value(transfer_function.get_value(
      network.neuron_array(neuron_index).transfer_function(), needed_input_dependency->get_value(0u/*past_index*/)
    ));
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ) override{
    parameter_not_used(network_input);
    parameter_not_used(label_data);
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(needed_input_dependency));
    RFASSERT(needed_input_dependency->is_processed());
    set_derivative(d_w_index, transfer_function.get_derivative( /* d t(f(w))/dx = f'(w) * t'(f(w))*/
      network.neuron_array(neuron_index).transfer_function(),
      needed_input_dependency->get_value(0u/*past_index*/),
      needed_input_dependency->get_derivative(0u/*past_index*/, d_w_index)
    ));
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return "";
  }

  std::string value_kernel_operation(
    std::string /*network_input_array*/, std::string /*weight_array*/,
    std::string operations_value_array, std::string /*operations_array_size*/
  ) const override{
    RFASSERT(static_cast<bool>(needed_input_dependency));
    RFASSERT(needed_input_dependency->are_dependencies_registered());
    return ( operations_value_array + "[" + std::to_string(get_operation_index()) + "] = "
      + transfer_function.get_kernel_function_for(
        network.neuron_array(neuron_index).transfer_function(),
        operations_value_array + "[" + std::to_string(needed_input_dependency->get_operation_index()) + "]"
      )
    ) + ";";
  }

  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*label_array*/, std::string /*weight_array*/,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string /*operations_array_size*/, std::string /*d_operations_array_size*/
  ) const override{
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(needed_input_dependency));
    return (
      operations_derivative_array + "[" + std::to_string(get_operation_index()) + "] = "
      + transfer_function.get_kernel_function_for_d(
        network.neuron_array(neuron_index).transfer_function(),
        operations_value_array + "[" + std::to_string(needed_input_dependency->get_operation_index()) + "]",
        operations_derivative_array + "[" + std::to_string(needed_input_dependency->get_operation_index()) + "]"
      )
    ) + ";";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    RFASSERT(static_cast<bool>(needed_input_dependency));
    return {needed_input_dependency};
  }

private:
  const rafko_net::TransferFunction transfer_function;
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> needed_input_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_TRANSFER_FN_OPERATION_H */
