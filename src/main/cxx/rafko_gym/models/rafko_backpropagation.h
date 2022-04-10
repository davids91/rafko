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

#ifndef RAFKO_BACKPROPAGATION_H
#define RAFKO_BACKPROPAGATION_H

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <map>

#include "rafko_utilities/services/rafko_math_utils.h"
#include "rafko_mainframe/models/rafko_settings.h"

#include "rafko_gym/models/rafko_backpropagation_operation.h"
#include "rafko_gym/models/rafko_backprop_network_input_operation.h"
#include "rafko_gym/models/rafko_backprop_neuron_input_operation.h"
#include "rafko_gym/models/rafko_backprop_transfer_fn_operation.h"
#include "rafko_gym/models/rafko_backprop_spike_fn_operation.h"
#include "rafko_gym/models/rafko_backprop_objective_operation.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackPropagation{
public:
  RafkoBackPropagation(const rafko_net::RafkoNet& network, rafko_mainframe::RafkoSettings& settings_)
  : settings(settings_)
  , network(network_)
  {
    network_input_derivatives.reserve(network.input_data_size())
  }

  void build(RafkoEnvironment& environment, RafkoObjective& objective){
    for(std::uint32_t output_index = 0; output_index < network.output_neuron_number(); ++output_index){
      (void)push_dependency(
        ad_operation_objective_d, *this, network,
        objective, output_index, environment.get_number_of_label_samples()
      );
    }

    std::uint32_t done_index = 0;
    while(done_index < operations.size()){
      if(!operations[done_index]->are_dependencies_registered()){
        operations[done_index]->upload_dependencies_to_operations();
      }
      ++done_index;
    }
  }

  //TODO: store values and weight derivatives for each run in the vectors
  //TODO: Use one buffer for calculated values, and another for calculated derivatives
  template<typename Args...>
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(
    Autodiff_operations type, const rafko_net::RafkoNet& network_,
    std::uint32_t past_index, std::uint32_t content_index /* neuron index or label index or input index */,
    Args... arguments
  ){
    switch(type){
      case ad_operation_objective_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropObjectiveOperation>(
          *this, network, past_index, content_index, arguments...
        ));
      case ad_operation_neuron_spike_d:
        return find_or_add_spike(past_index, content_index, arguments...);
      case ad_operation_neuron_transfer_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropTransferFnOperation>(
          *this, network, past_index, content_index, settings, arguments...
        ));
      case ad_operation_neuron_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNeuronInputOperation>(
          *this, network, past_index, content_index, arguments...
        ));
      case ad_operation_network_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNetworkInputOperation>(
          *this, network, past_index, content_index, arguments...
        ));
      }break;
    }
  }

private:
  const rafko_net::RafkoNet& network;
  rafko_mainframe::RafkoSettings& settings;
  std::map<std::uint64_t, std::uint32_t> neuron_and_past_to_operation_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;
  std::vector<std::vector<double>> calculated_values; /* {runs,operations} */
  std::vector<std::vector<double>> /* {runs, weights} */

  template<typename Args...>
  static std::shared_ptr<RafkoBackpropagationOperation> find_or_add_spike(
    std::uint32 past_index, std::uint32_t neuron_index, Args... arguments
  ){
    auto found_element = neuron_and_past_to_operation_map.find(
      rafko_utils::pair_hash({past_index, neuron_index})
    );
    if(found_element != neuron_and_past_to_operation_map.end())
      return found_element->second;
    operations.emplace_back(std::make_shared<RafkoBackpropSpikeFnOperation>(
      *this, network, past_index, neuron_index, arguments...
    ));
    return std::get<0>(neuron_and_past_to_operation_map.insert(
      {rafko_utils::pair_hash({past_index, neuron_index}), (operations.size() - 1u)}
    ));
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_H */
