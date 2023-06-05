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
#include "rafko_gym/services/rafko_backprop_weight_reg_operation.hpp"

namespace rafko_gym {

void RafkoBackpropWeightRegOperation::refresh_weight_derivatives() {
  m_relevantIndexValues.clear();
  rafko_net::SynapseIterator<>::iterate(
      m_featureGroup.relevant_neurons(), [this](std::uint32_t neuron_index) {
        rafko_net::SynapseIterator<>::iterate(
            m_network.neuron_array(neuron_index).input_weights(),
            [this](std::uint32_t weight_index) {
              m_relevantIndexValues.push_back(weight_index);
              if (m_featureGroup.feature() ==
                  rafko_net::neuron_group_feature_l1_regularization) {
                m_eachWeightDerivative[weight_index] = 1.0;
              } else if (m_featureGroup.feature() ==
                         rafko_net::neuron_group_feature_l2_regularization) {
                m_eachWeightDerivative[weight_index] =
                    2.0 * m_network.weight_table(weight_index);
              }
            });
      });
}

} /* namespace rafko_gym */
