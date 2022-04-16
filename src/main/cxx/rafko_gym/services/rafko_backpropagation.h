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
  , calculated_derivatives(network.memory_size())
  , calculated_values(network.memory_size())
  {
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

    calculated_values = std::vector<std::vector<double>>(
      network.memory_size(), std::vector<double>(operations.size())
    );
    for(std::vector<std::vector<double>>& past : calculated_derivatives){
      for(std::vector<std::vector<double>>& operation : past){
        operation = std::vector<double>(network.weight_table_size());
      }
    }
  }

private:
  const rafko_net::RafkoNet& network;
  rafko_mainframe::RafkoSettings& settings;
  std::vector<std::vector<std::vector<double>>> calculated_derivatives; /* {runs, operations, d_w values} */
  std::vector<std::vector<double>> calculated_values; /* {runs, operations} */
  std::map<std::uint32_t, std::uint32_t> neuron_spike_to_operation_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;

  void set_value(std::uint32_t run_index, std::uint32_t operation_index, double value){
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(operation_index < calculated_values[run_index].size());
    calculated_values[run_index][operation_index] = value;
  }

  // TODO:constexpr these
  void set_derivative(
    std::uint32_t run_index, std::uint32_t operation_index,
    std::uint32_t d_w_index, double value
  ){
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(d_w_index < calculated_values[run_index].size());
    calculated_derivatives[run_index][operation_index][d_w_index] = value;
  }

  void get_value(std::uint32_t run_index, std::uint32_t operation_index){
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(operation_index < calculated_values[run_index].size());
    return calculated_values[run_index][operation_index];
  }

  void get_derivative(std::uint32_t run_index, std::uint32_t operation_index, std::uint32_t weight_index){
    RFASSERT(run_index < calculated_values.size());
    RFASSERT(operation_index < calculated_values[run_index].size());
    RFASSERT(weight_index < calculated_values[run_index][operation_index].size());
    return calculated_derivatives[run_index][operation_index][weight_index];
  }

  //TODO: Remove past index ??
  template<typename Args...>
  static std::shared_ptr<RafkoBackpropagationOperation> find_or_add_spike(std::uint32_t neuron_index, Args... arguments){
    auto found_element = neuron_spike_to_operation_map.find(neuron_index);
    if(found_element != neuron_spike_to_operation_map.end())
      return operations[found_element->second];

    operations.emplace_back(std::make_shared<RafkoBackpropSpikeFnOperation>(
      *this, network, operations.size(), neuron_index, arguments...
    ));
    neuron_spike_to_operation_map.insert( {neuron_index, (operations.size() - 1u)} );
    return operations.back();
  }

  //TODO: store values and weight derivatives for each run in the vectors
  //TODO: Use one buffer for calculated values, and another for calculated derivatives
  template<typename Args...>
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency( /* neuron index or label index or input index */
    Autodiff_operations type, std::uint32_t content_index, Args... arguments
  ){
    switch(type){
      case ad_operation_objective_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropObjectiveOperation>(
          *this, network, operations.size(), content_index, arguments...
        ));
      case ad_operation_neuron_spike_d:
        return find_or_add_spike(content_index, arguments...);
      case ad_operation_neuron_transfer_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropTransferFnOperation>(
          *this, network, operations.size(), content_index, settings, arguments...
        ));
      case ad_operation_neuron_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNeuronInputOperation>(
          *this, network, operations.size(), content_index, arguments...
        ));
      case ad_operation_network_input_d:
        return operations.emplace_back(std::make_shared<RafkoBackpropNetworkInputOperation>(
          *this, network, operations.size(), content_index, arguments...
        ));
      }break;
    }
  }
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_H */
