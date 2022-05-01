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

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <utility>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/models/subscript_proxy.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"
#include "rafko_net/services/rafko_network_feature.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackpropFeatureOperation
: public RafkoBackpropagationOperation
{
public:
  RafkoBackpropFeatureOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index,  const rafko_mainframe::RafkoSettings& settings_,
    const rafko_net::FeatureGroup& feature_group_,
    std::shared_ptr<rafko_utilities::SubscriptDictionary> neuron_index_dictionary
  )
  : RafkoBackpropagationOperation(data, network, operation_index)
  , settings(settings_)
  , feature_group(feature_group_)
  , network_data_proxy(dummy_vector, neuron_index_dictionary)
  , execution_threads()
  , feature_executor(execution_threads)
  {
    for(std::uint32_t thread_index = 0; thread_index < settings.get_max_processing_threads(); ++ thread_index)
      execution_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_solve_threads()));
  }

  DependencyRequest upload_dependencies_to_operations();

  void calculate_value(const std::vector<double>& network_input){
    parameter_not_used(network_input);
    network_data_proxy = network_data_proxy.create(
      data.get_mutable_value().get_element(0)
    );
    feature_executor.execute_solution_relevant(
      feature_group, settings, network_data_proxy, 0u/*thread_index*/
    );
    set_value_processed();
  }

  void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ){
    parameter_not_used(d_w_index);
    parameter_not_used(network_input);
    parameter_not_used(label_data);
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
  const rafko_mainframe::RafkoSettings& settings;
  const rafko_net::FeatureGroup& feature_group;
  rafko_utilities::SubscriptProxy<> network_data_proxy;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> execution_threads;
  rafko_net::RafkoNetworkFeature feature_executor;

  std::vector<double> dummy_vector;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROP_FEATURE_OPERATION_H */
