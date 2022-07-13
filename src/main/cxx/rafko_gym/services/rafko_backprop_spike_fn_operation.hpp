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

#ifndef RAFKO_BACKPROP_SPIKE_FN_OPERATION_H
#define RAFKO_BACKPROP_SPIKE_FN_OPERATION_H

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>

#if(RAFKO_USES_OPENCL)
#include <string>
#include <regex>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/models/spike_function.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropSpikeFnOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropSpikeFnOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t neuron_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_neuron_spike_d)
  , neuron_index(neuron_index_)
  , actual_operation_index(operation_index)
  {
  }
  ~RafkoBackpropSpikeFnOperation() = default;

  std::uint32_t get_operation_index() const override{
    return actual_operation_index;
  }

  void set_operation_index(std::uint32_t index){
    RFASSERT(!operation_index_final);
    actual_operation_index = index;
  }

  bool operation_index_finalised() override{
    return operation_index_final;
  }

  DependencyRequest upload_dependencies_to_operations() override{
    return {{
      {{ad_operation_neuron_transfer_d, {neuron_index}}},
      [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>> dependencies){
        RFASSERT(1 == dependencies.size());
        present_value_dependency = dependencies[0];
        set_registered();
      }
    }};
  }

  void calculate_value(const std::vector<double>& network_input) override;

  void calculate_derivative(
    std::uint32_t d_w_index,
    const std::vector<double>& network_input, const std::vector<double>& label_data
  ) override;

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override;
  std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size
  ) const override;
  std::string derivative_kernel_operation(
    std::string network_input_array, std::string label_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, std::string d_operations_array_size
  ) const override;
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    RFASSERT(static_cast<bool>(present_value_dependency));
    return {present_value_dependency};
  }

private:
  const std::uint32_t neuron_index;
  std::shared_ptr<RafkoBackpropagationOperation> present_value_dependency;
  std::uint32_t actual_operation_index;
  bool operation_index_final = false;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_SPIKE_FN_OPERATION_H */
