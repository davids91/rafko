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
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    const RafkoObjective& objective_, std::uint32_t operation_index,
    std::uint32_t output_index_,  std::uint32_t sample_number_
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_objective_d)
  , objective(objective_)
  , output_index(output_index_)
  , sample_number(sample_number_)
  {
  }
  ~RafkoBackpropObjectiveOperation() = default;

  DependencyRequest upload_dependencies_to_operations() override{
    return {{
      {{ad_operation_neuron_spike_d, {(network.neuron_array_size() - network.output_neuron_number() + output_index)}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        feature_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate_value(const std::vector<double>& network_input) override{
    parameter_not_used(network_input);
    RFASSERT(are_dependencies_registered());
    RFASSERT(static_cast<bool>(feature_dependency));
    RFASSERT(feature_dependency->is_value_processed());
    /*!Note: Value is not being calculated, because they are not of use (as of now.. ) */
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ) override{
    parameter_not_used(network_input);
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(output_index < label_data.size());
    RFASSERT(static_cast<bool>(feature_dependency));
    RFASSERT(feature_dependency->is_processed());
    set_derivative(d_w_index, objective.get_derivative(
      label_data[output_index], feature_dependency->get_value(0u/*past_index*/),
      feature_dependency->get_derivative(0u/*past_index*/, d_w_index), static_cast<double>(sample_number)
    ));
    RFASSERT_LOG(
      "derivative operation[{}](w[{}]): Objective[{}]_d = {} = derivative({}(label[{}]),{}(op[{}]),{}(d_op),{}(samples))",
      get_operation_index(), d_w_index, output_index, get_derivative(0u/*past_index*/, d_w_index),
      label_data[output_index], output_index,
      feature_dependency->get_value(0u/*past_index*/), feature_dependency->get_operation_index(),
      feature_dependency->get_derivative(0u/*past_index*/, d_w_index), static_cast<double>(sample_number)
    );
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return "";
  }


  std::string value_kernel_operation(
    std::string /*network_input_array*/, std::string /*weight_array*/,
    std::string /*operations_value_array*/, std::string /*operations_array_size*/
  ) const override{ /*!Note: Value is not being calculated, because they are not of use (as of now.. ) */
    return "";
  }
  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string label_array, std::string /*weight_array*/,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string /*operations_array_size*/, std::string /*d_operations_array_size*/
  ) const override{
    RFASSERT(static_cast<bool>(feature_dependency));
    return (
      operations_derivative_array + "[" + std::to_string(get_operation_index()) + "] = "
      + objective.get_derivative_kernel_source(
        label_array + "[" + std::to_string(output_index) + "]",
        operations_value_array + "[" + std::to_string(feature_dependency->get_operation_index()) + "]",
        operations_derivative_array + "[" + std::to_string(feature_dependency->get_operation_index()) + "]",
        std::to_string(sample_number)
      ) + ";"
    );
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    RFASSERT(static_cast<bool>(feature_dependency));
    return {feature_dependency};
  }

private:
  const RafkoObjective& objective;
  const std::uint32_t output_index;
  const std::uint32_t sample_number;
  std::shared_ptr<RafkoBackpropagationOperation> feature_dependency;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_OBJECTIVE_OPERATION_H */
