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

#ifndef RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H
#define RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropNetworkInputOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropNetworkInputOperation(
    RafkoBackPropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t input_index_, std::uint32_t weight_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , input_index(input_index_)
  , weight_index(weight_index_)
  {
  }

  DependencyRequest upload_dependencies_to_operations(){
    /*!Note: Network inputs have no dependencies! */
    set_registered();
    return {};
  }

  void calculate(
    std::uint32_t d_w_index, std::uint32_t run_index,
    const std::vector<std::vector<double>>& network_input, const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(run_index < network_input.size());
    RFASSERT(run_index < label_data.size());
    set_value(run_index, network_input[run_index][input_index] * network.weight_table(weight_index));
    set_derivative(
      run_index, d_w_index, ((d_w_index == weight_index)?(network_input[run_index][input_index]):(0.0))
    );
    set_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function() const{
    return (
      " input[" +std::to_string(input_index) + "]"
      + " * weight[" + std::to_string(weight_index) + "]"
      + "(" + std::to_string(network.weight_table(weight_index)) + ")"
    );
  }
  std::string derivative_kernel_function() const{
    return "";
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_dependencies(){
    return {};
  }

private:
  const std::uint32_t input_index;
  const std::uint32_t weight_index;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H */
