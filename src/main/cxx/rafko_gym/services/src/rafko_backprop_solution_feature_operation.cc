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
#include "rafko_gym/services/rafko_backprop_solution_feature_operation.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

RafkoBackPropSolutionFeatureOperation::RafkoBackPropSolutionFeatureOperation(
  RafkoBackpropagationData& data, const rafko_net::RafkoNet& network_,
  std::uint32_t operation_index,  const rafko_mainframe::RafkoSettings& settings_,
  const rafko_net::FeatureGroup& feature_group_,
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads_,
  std::shared_ptr<rafko_utilities::SubscriptDictionary> neuron_index_dictionary
)
: RafkoBackpropagationOperation(data, network_, operation_index, ad_operation_network_feature)
, settings(settings_)
, network(network_)
, feature_group(feature_group_)
, network_data_proxy(dummy_vector, neuron_index_dictionary)
, execution_threads(execution_threads_)
, feature_executor(execution_threads)
{
  #if(RAFKO_USES_OPENCL)
  /* Calculate relevant index values */
  switch(feature_group.feature()){
    case rafko_net::neuron_group_feature_softmax:
    case rafko_net::neuron_group_feature_dropout_regularization:{
      rafko_net::SynapseIterator<>::iterate(feature_group.relevant_neurons(),[this](std::uint32_t index){
        relevant_index_values.push_back(index);
      });
    } break;
    case rafko_net::neuron_group_feature_l1_regularization:
    case rafko_net::neuron_group_feature_l2_regularization: {
      rafko_net::SynapseIterator<>::iterate(feature_group.relevant_neurons(), [this](std::uint32_t neuron_index){
        rafko_net::SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
        [this](std::uint32_t weight_index){
          relevant_index_values.push_back(weight_index);
        });
      });
    } break;
    default: break; /* unknown functionality should yield no relevant index values */
  }
  #endif/*(RAFKO_USES_OPENCL)*/
}

DependencyRequest RafkoBackPropSolutionFeatureOperation::upload_dependencies_to_operations(){
  DependencyParameters dependency_parameters;
  rafko_net::SynapseIterator<>::iterate(feature_group.relevant_neurons(),
  [&dependency_parameters](std::uint32_t neuron_index){
    dependency_parameters.push_back({ ad_operation_neuron_spike_d, { neuron_index } });
  });

  set_registered();
  return {{
    dependency_parameters, [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>){}
  }};
}


} /* namespace rafko_gym */
