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

#include "rafko_utilities/services/rafko_string_utils.h"
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
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t input_index_, std::uint32_t weight_index_
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_network_input_d)
  , input_index(input_index_)
  , weight_index(weight_index_)
  {
  }
  ~RafkoBackpropNetworkInputOperation() = default;

  DependencyRequest upload_dependencies_to_operations(){
    /*!Note: Network inputs have no dependencies! */
    set_registered();
    return {};
  }

  void calculate_value(const std::vector<double>& network_input){
    RFASSERT(input_index < network_input.size());
    set_value(network_input[input_index] * network.weight_table(weight_index));
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ){
    parameter_not_used(label_data);
    RFASSERT(input_index < network_input.size());
    set_derivative( d_w_index, ((d_w_index == weight_index)?(network_input[input_index]):(0.0)) );
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const{
    return "";
  }


  std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string /*operations_array_size*/
  ) const{
    return (
      operations_value_array + "[" + std::to_string(get_operation_index()) + "] = "
      + network_input_array + "[" + std::to_string(input_index) + "]"
      + " * " + weight_array + "[" + std::to_string(weight_index) + "];\n"
    );
  }
  //TODO: d_w_index, available_memory_slots inside the kernels are hidden dependencies!
  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*label_array*/, std::string weight_array,
    std::string /*operations_value_array*/, std::string operations_derivative_array,
    std::string /*operations_array_size*/
  ) const{
    std::string kernel_code = R"(
      if(d_w_index == ==this_op_weight_index==){
        ==op_derivative_array==[==op_index==] = (
          ==weight_value==
        );
      }else{
        ==op_derivative_array==[==op_index==] = 0.0;
      }
    )";
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==this_op_weight_index=="), std::to_string(weight_index)
    );
    kernel_code = rafko_utilities::replace_all_in_string(
      kernel_code, std::regex("==weight_value=="),
      weight_array + "[" + std::to_string(weight_index) + "]"
    );
    kernel_code = rafko_utilities::replace_all_in_string(kernel_code, std::regex("==op_derivative_array=="), operations_derivative_array);
    kernel_code = rafko_utilities::replace_all_in_string(kernel_code, std::regex("==op_index=="), std::to_string(get_operation_index()));
    return kernel_code;
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
