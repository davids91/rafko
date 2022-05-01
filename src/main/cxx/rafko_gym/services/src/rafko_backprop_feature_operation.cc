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
#include "rafko_gym/services/rafko_backprop_feature_operation.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

DependencyRequest RafkoBackpropFeatureOperation::upload_dependencies_to_operations(){
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
