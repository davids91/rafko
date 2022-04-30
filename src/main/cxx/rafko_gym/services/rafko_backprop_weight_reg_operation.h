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

#ifndef RAFKO_BACKPROP_WEIGHT_REG_OPERATION_H
#define RAFKO_BACKPROP_WEIGHT_REG_OPERATION_H

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_network_feature.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropWeightRegOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropWeightRegOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, const rafko_net::FeatureGroup& feature_group_
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , feature_group(feature_group_)
  , each_weight_derivative(network.weight_table_size())
  {
    refresh_weight_derivatives();
  }

  DependencyRequest upload_dependencies_to_operations(){
    set_registered();
    return {};
  }

  void calculate_value(const std::vector<double>& network_input){
    parameter_not_used(network_input);
    /*!Note: Calculated value is not exactly important here, but avg_derivatives
     * need only be calculated once for weight regularization logic, so they are calculated here
     */
    if(feature_group.feature() == rafko_net::neuron_group_feature_l2_regularization){
      refresh_weight_derivatives();
    }
    /*!Note: l1 need not be refreshed, as structural changes are not yet present */
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ){
    parameter_not_used(network_input);
    parameter_not_used(label_data);
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(d_w_index < each_weight_derivative.size());
    RFASSERT(static_cast<std::int32_t>(d_w_index) < network.weight_table_size());
    set_derivative(d_w_index, each_weight_derivative[d_w_index]);
    set_derivative_processed();
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
    return {};
  }

private:
  const rafko_net::FeatureGroup& feature_group;
  std::vector<double> each_weight_derivative;

  void refresh_weight_derivatives(){
    rafko_net::SynapseIterator<>::iterate(feature_group.relevant_neurons(),
    [this](std::uint32_t neuron_index){
      rafko_net::SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
      [this](std::uint32_t weight_index){
        if(feature_group.feature() == rafko_net::neuron_group_feature_l1_regularization){
          each_weight_derivative[weight_index] = 1.0;
        }else if(feature_group.feature() == rafko_net::neuron_group_feature_l2_regularization){
          each_weight_derivative[weight_index] = 2.0 * network.weight_table(weight_index);
        }
      });
    });
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_WEIGHT_REG_OPERATION_H */
