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

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>
#if(RAFKO_USES_OPENCL)
#include <string>

#include "rafko_utilities/services/rafko_string_utils.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropNetworkInputOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropNetworkInputOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, std::uint32_t input_index, std::uint32_t weight_index
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_network_input_d)
  , m_inputIndex(input_index)
  , m_weightIndex(weight_index)
  {
  }
  ~RafkoBackpropNetworkInputOperation() = default;

  std::uint32_t get_weight_index() const{
    return m_weightIndex;
  }

  std::uint32_t get_input_index() const{
    return m_inputIndex;
  }

  DependencyRequest upload_dependencies_to_operations() override{
    /*!Note: Network inputs have no dependencies! */
    set_registered();
    return {};
  }

  void calculate_value(const std::vector<double>& network_input) override{
    RFASSERT(m_inputIndex < network_input.size());
    set_value(network_input[m_inputIndex] * m_network.weight_table(m_weightIndex));
    RFASSERT_LOG(
      "operation[{}]: Network Input[{}]({}) * weight[{}]({}) = {}", get_operation_index(),
      m_inputIndex, network_input[m_inputIndex], m_weightIndex, m_network.weight_table(m_weightIndex),
      ( network_input[m_inputIndex] * m_network.weight_table(m_weightIndex) )
    );
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& /*label_data*/
  ) override{
    RFASSERT(m_inputIndex < network_input.size());
    set_derivative( d_w_index, ((d_w_index == m_weightIndex)?(network_input[m_inputIndex]):(0.0)) );
    RFASSERT_LOG(
      "derivative operation[{}](w[{}]): Network Input[{}]_d = {}", get_operation_index(),
      d_w_index, m_inputIndex, get_derivative(0u/*past_index*/, d_w_index)
    );
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return "";
  }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for forward propagation
   * 
   * @param   network_input_array           The name of the arry containing the Inputs for the Neural network
   * @param   weight_array                  The name of the array contining the Neural network weights 
   * @param   operations_value_array        The name of the array containing the operation values for forward propagation
   *
   * @return    Raw Kernel code for the forward propagation of this operation
   */
  static std::string generic_value_kernel_operation(std::string network_input_array, std::string weight_array, std::string operations_value_array){
    return (
      operations_value_array + "[==op_index==] = " + network_input_array + "[==network_input_index==]"
      + " * " + weight_array + "[==this_op_weight_index==];\n"
    );
  }

  std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string /*operations_array_size*/
  ) const override{
    std::string kernel_source = generic_value_kernel_operation(network_input_array, weight_array, operations_value_array);
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_input_index=="), std::to_string(m_inputIndex)
    );
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==this_op_weight_index=="), std::to_string(m_weightIndex)
    );
    kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_index=="), std::to_string(get_operation_index()));
    return kernel_source;
  }

  /**
   * @brief     Generates OpenCL Kernel code for the operation for backward propagation
   * 
   * @param   network_input_array           The name of the arry containing the Inputs for the Neural network
   * @param   operations_derivative_array   The name of the array containing the operation values for backward propagation
   *
   * @return    Raw Kernel code for the backward propagation of this operation
   */
  static std::string generic_derivative_kernel_operation(std::string network_input_array, std::string operations_derivative_array){
    std::string kernel_source = R"(
      if(d_w_index == ==this_op_weight_index==){
        ==op_derivative_array==[==op_index==] = ( ==network_input_array==[==network_input_index==] );
      }else{
        ==op_derivative_array==[==op_index==] = 0.0;
      }
    )";
    kernel_source = rafko_utilities::replace_all_in_string(
      kernel_source, std::regex("==network_input_array=="), network_input_array
    );
    kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==op_derivative_array=="), operations_derivative_array);
    return kernel_source;
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    return {};
  }

private:
  const std::uint32_t m_inputIndex;
  const std::uint32_t m_weightIndex;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_NETWORK_INPUT_OPERATION_H */
