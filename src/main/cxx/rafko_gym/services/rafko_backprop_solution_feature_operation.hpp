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

#ifndef RAFKO_BACKPROP_FEATURE_OPERATION_H
#define RAFKO_BACKPROP_FEATURE_OPERATION_H

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <utility>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/models/subscript_proxy.hpp"
#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/services/rafko_network_feature.hpp"
#include "rafko_net/models/neuron_info.hpp"

#include "rafko_gym/services/rafko_backpropagation_operation.hpp"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_EXPORT RafkoBackPropSolutionFeatureOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackPropSolutionFeatureOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index,  const rafko_mainframe::RafkoSettings& settings,
    const rafko_net::FeatureGroup& feature_group,
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads,
    std::shared_ptr<rafko_utilities::SubscriptDictionary> neuron_index_dictionary
  );
  ~RafkoBackPropSolutionFeatureOperation() = default;

  DependencyRequest upload_dependencies_to_operations() override;

  void calculate_value(const std::vector<double>& /*network_input*/) override;
  void calculate_derivative(
    std::uint32_t /*d_w_index*/, const std::vector<double>& /*network_input*/, const std::vector<double>& /*label_data*/
  ) override{
    set_derivative_processed();
  }

  #if(RAFKO_USES_OPENCL)
  std::string local_declaration_operation() const override{
    return rafko_net::RafkoNetworkFeature::get_kernel_locals();
  }

  std::string value_kernel_operation(
    std::string network_input_array, std::string /*weight_array*/,
    std::string operations_value_array, std::string /*operations_array_size*/
  ) const override;

  std::string derivative_kernel_operation(
    std::string /*network_input_array*/, std::string /*label_array*/, std::string /*weight_array*/,
    std::string /*operations_value_array*/, std::string /*operations_derivative_array*/,
    std::string /*operations_array_size*/, std::string /*d_operations_array_size*/
  ) const override{ /*!Note: solution features don't have any derivatives, so nothing is here.. */
    return "";
  }
  bool is_multi_worker() const override{
    return true;
  }
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> get_own_dependencies() override{
    return {};
  }

private:
  const rafko_mainframe::RafkoSettings& m_settings;
  const rafko_net::FeatureGroup& m_featureGroup;
  rafko_utilities::SubscriptProxy<> m_networkDataProxy;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& m_executionThreads;
  rafko_net::RafkoNetworkFeature m_featureExecutor;
  std::vector<std::uint32_t> m_relevantIndexValues;
  #if(RAFKO_USES_OPENCL)
  std::shared_ptr<rafko_utilities::SubscriptDictionary> m_neuronIndexDictionary;
  #endif/*(RAFKO_USES_OPENCL)*/

  std::vector<double> m_dummyVector;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_FEATURE_OPERATION_H */
