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
#include <utility>
#include <map>

#include "rafko_mainframe/models/rafko_settings.h"

#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_backpropagation_data.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"
#include "rafko_gym/services/rafko_backprop_network_input_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.h"
#include "rafko_gym/services/rafko_backprop_transfer_fn_operation.h"
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.h"
#include "rafko_gym/services/rafko_backprop_objective_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoBackPropagation{
public:
  RafkoBackPropagation(const rafko_net::RafkoNet& network_, rafko_mainframe::RafkoSettings& settings_)
  : settings(settings_)
  , network(network_)
  , data(network)
  {
  }

  void build(RafkoEnvironment& environment, RafkoObjective& objective){
    for(std::uint32_t output_index = 0; output_index < network.output_neuron_number(); ++output_index){
      operations.emplace_back(std::make_shared<RafkoBackpropObjectiveOperation>(
        data, network, objective, operations.size(), output_index, environment.get_number_of_label_samples()
      ));
    }

    std::uint32_t done_index = 0;
    while(done_index < operations.size()){
      if(!operations[done_index]->are_dependencies_registered()){
        DependencyRequest request = operations[done_index]->upload_dependencies_to_operations();
        if(request.has_value()){
          auto& [parameters, dependency_register] = request.value();
          std::vector<std::shared_ptr<RafkoBackpropagationOperation>> new_dependencies;
          for(const DependencyParameter& parameter : parameters)
            new_dependencies.push_back(push_dependency(parameter));
          dependency_register(new_dependencies);
        }
      }
      ++done_index;
    }/*while(done_index < operations.size())*/
    data.build(operations.size());
  }

  void calculate(
    const std::vector<std::vector<double>>& network_input,
    const std::vector<std::vector<double>>& label_data
  ){
    RFASSERT(network_input.size() == label_data.size());
    for(std::uint32_t run_index = 0; run_index < network_input.size(); ++run_index){
      RFASSERT(network_input[run_index].size() == label_data[run_index].size());
      for(std::int32_t weight_index = 0u; weight_index < network.weight_table_size(); ++weight_index)
        for(std::int32_t operation_index = operations.size() - 1; operation_index >= 0; --operation_index)
          operations[operation_index]->calculate(static_cast<std::uint32_t>(weight_index), run_index, network_input, label_data);
    }
  }

private:
  rafko_mainframe::RafkoSettings& settings;
  const rafko_net::RafkoNet& network;
  RafkoBackPropagationData data;
  std::map<std::uint32_t, std::uint32_t> neuron_spike_to_operation_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;

  std::shared_ptr<RafkoBackpropagationOperation> find_or_add_spike(std::uint32_t neuron_index){
    auto found_element = neuron_spike_to_operation_map.find(neuron_index);
    if(found_element != neuron_spike_to_operation_map.end())
      return operations[found_element->second];

    operations.emplace_back(new RafkoBackpropSpikeFnOperation(
      data, network, operations.size(), neuron_index
    ));
    neuron_spike_to_operation_map.insert( {neuron_index, (operations.size() - 1u)} );
    return operations.back();
  }

  using DependencyParameter = std::pair<Autodiff_operations,std::vector<std::uint32_t>>;
  using DependencyParameters = std::vector<DependencyParameter>;
  using DependencyRegister = std::function<void(std::vector<std::shared_ptr<RafkoBackpropagationOperation>>)>;
  using DependencyRequest = std::optional<std::pair<DependencyParameters,DependencyRegister>>;
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(DependencyParameter arguments){
    switch(std::get<0>(arguments)){
      case ad_operation_neuron_spike_d:
        RFASSERT(1u == std::get<1>(arguments).size());
        return find_or_add_spike(std::get<1>(arguments)[0]);
      case ad_operation_neuron_transfer_d:
        RFASSERT(1u == std::get<1>(arguments).size());
        return operations.emplace_back(new RafkoBackpropTransferFnOperation(
          data, network, operations.size(), std::get<1>(arguments)[0], settings
        ));
      case ad_operation_neuron_input_d:
        RFASSERT(2u == std::get<1>(arguments).size());
        return operations.emplace_back(new RafkoBackpropNeuronInputOperation(
          data, network, operations.size(), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
        ));
      case ad_operation_network_input_d:
        RFASSERT(2u == std::get<1>(arguments).size());
        return operations.emplace_back(new RafkoBackpropNetworkInputOperation(
          data, network, operations.size(), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
        ));
      break;
      case ad_operation_objective_d: /* Objective operations are placed manually to the beginning of the vector */
      case ad_operation_unknown:
      default: break;
    }
    return std::shared_ptr<RafkoBackpropagationOperation>();
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_H */
