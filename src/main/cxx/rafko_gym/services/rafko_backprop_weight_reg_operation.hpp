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

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/services/rafko_network_feature.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackpropWeightRegOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropWeightRegOperation(
    const rafko_mainframe::RafkoSettings& settings, RafkoBackpropagationData& data,
    const rafko_net::RafkoNet& network, std::uint32_t operation_index,
    const rafko_net::FeatureGroup& feature_group
  )
  : RafkoBackpropagationOperation(data, network, operation_index, ad_operation_network_weight_regularization_feature)
  , m_settings(settings)
  , m_featureGroup(feature_group)
  , m_eachWeightDerivative(m_network.weight_table_size())
  {
    m_relevantIndexValues.reserve(m_network.weight_table_size());
    refresh_weight_derivatives();
  }
  ~RafkoBackpropWeightRegOperation() = default;

  DependencyRequest upload_dependencies_to_operations() override{
    set_registered();
    return {};
  }

  void calculate_value(const std::vector<double>& /*network_input*/) override{
    /*!Note: Calculated value is not exactly important here, but avg_derivatives
     * need only be calculated once for weight regularization logic, so they are calculated here
     */
    if(m_featureGroup.feature() == rafko_net::neuron_group_feature_l2_regularization){
      refresh_weight_derivatives();
    }
    /*!Note: l1 need not be refreshed, as structural changes are not yet present */
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& /*network_input*/, const std::vector<double>& /*label_data*/
  ) override{
    RFASSERT(is_value_processed());
    RFASSERT(are_dependencies_registered());
    RFASSERT(d_w_index < m_eachWeightDerivative.size());
    RFASSERT(static_cast<std::int32_t>(d_w_index) < m_network.weight_table_size());
    set_derivative(d_w_index, m_eachWeightDerivative[d_w_index]);
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return rafko_net::RafkoNetworkFeature::get_kernel_locals();
  }

  std::string value_kernel_operation(
    std::string /*network_input_array*/, std::string /*weight_array*/,
    std::string /*operations_value_array*/, std::string /*operations_array_size*/
  ) const override{
    /*!Note: No actual value is calculated for weight regularization */
    return "";
  }

  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*label_array*/, std::string weight_array,
    std::string /*operations_value_array*/, std::string operations_derivative_array,
    std::string /*operations_array_size*/, std::string /*d_operations_array_size*/
  ) const override{
    return rafko_net::RafkoNetworkFeature::generate_kernel_code(
      m_settings, m_featureGroup.feature(), m_relevantIndexValues,
      weight_array, "0"/*input_start_index*/, operations_derivative_array/* output_array */,
      std::to_string(get_operation_index())/*output_start_index*/, false/*declare_locals*/
    );
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    return {};
  }

private:
  const rafko_mainframe::RafkoSettings& m_settings;
  const rafko_net::FeatureGroup& m_featureGroup;
  std::vector<double> m_eachWeightDerivative;
  std::vector<std::uint32_t> m_relevantIndexValues;

  void refresh_weight_derivatives();
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_WEIGHT_REG_OPERATION_H */
