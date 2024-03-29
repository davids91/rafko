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
#include "rafko_gym/services/rafko_backprop_solution_feature_operation.hpp"

#include "rafko_net/services/synapse_iterator.hpp"

namespace rafko_gym {

RafkoBackPropSolutionFeatureOperation::RafkoBackPropSolutionFeatureOperation(
    RafkoBackpropagationData &data, const rafko_net::RafkoNet &network,
    std::uint32_t operation_index,
    const rafko_mainframe::RafkoSettings &settings,
    const rafko_net::FeatureGroup &feature_group,
    rafko_utilities::SubscriptProxy<>::AssociationVector
        neuronSpikeToOperationIndex,
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>
        &execution_threads)
    : RafkoBackpropagationOperation(data, network, operation_index,
                                    ad_operation_network_feature),
      m_settings(settings), m_featureGroup(feature_group),
      m_networkDataProxy(m_dummyVector, neuronSpikeToOperationIndex),
      m_executionThreads(execution_threads),
      m_featureExecutor(m_executionThreads) {
#if (RAFKO_USES_OPENCL)
  /* Calculate relevant index values */
  switch (m_featureGroup.feature()) {
  case rafko_net::neuron_group_feature_softmax:
  case rafko_net::neuron_group_feature_dropout_regularization: {
    rafko_net::SynapseIterator<>::iterate(
        feature_group.relevant_neurons(), [this](std::uint32_t index) {
          m_relevantIndexValues.push_back(index);
        });
  } break;
  case rafko_net::neuron_group_feature_l1_regularization:
  case rafko_net::neuron_group_feature_l2_regularization: {
    rafko_net::SynapseIterator<>::iterate(
        feature_group.relevant_neurons(), [this](std::uint32_t neuron_index) {
          rafko_net::SynapseIterator<>::iterate(
              m_network.neuron_array(neuron_index).input_weights(),
              [this](std::uint32_t weight_index) {
                m_relevantIndexValues.push_back(weight_index);
              });
        });
  } break;
  default:
    break; /* unknown functionality should yield no relevant index values */
  }
#endif /*(RAFKO_USES_OPENCL)*/
}

void RafkoBackPropSolutionFeatureOperation::calculate_value(
    const std::vector<double> & /*network_input*/) {
  m_networkDataProxy.update(m_data.get_mutable_value().get_element(0));
  m_featureExecutor.execute_solution_relevant(
      m_featureGroup, m_settings, m_networkDataProxy, 0u /*thread_index*/
  );
  set_value_processed();
}

RafkoBackpropagationOperation::DependencyRequest
RafkoBackPropSolutionFeatureOperation::request_dependencies() {
  RafkoBackpropagationOperation::DependencyParameters dependency_parameters;
  rafko_net::SynapseIterator<>::iterate(
      m_featureGroup.relevant_neurons(),
      [&dependency_parameters](std::uint32_t neuron_index) {
        dependency_parameters.push_back(
            {ad_operation_neuron_spike_d, {neuron_index}});
      });

  set_registered();
  return {
      {dependency_parameters,
       [this](std::vector<std::shared_ptr<RafkoBackpropagationOperation>>) {}}};
}

} /* namespace rafko_gym */
