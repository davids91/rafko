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
#include "rafko_gym/services/rafko_autodiff_optimizer.h"

namespace rafko_gym{

void RafkoAutodiffOptimizer::build(RafkoEnvironment& environment, RafkoObjective& objective){
  RFASSERT_SCOPE(AUTODIFF_BUILD);
  for(std::uint32_t output_index = 0; output_index < network.output_neuron_number(); ++output_index){
    operations.emplace_back(std::make_shared<RafkoBackpropObjectiveOperation>(
      data, network, objective, operations.size(),
      output_index, environment.get_number_of_label_samples()
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

void RafkoAutodiffOptimizer::calculate(
  const std::vector<std::vector<double>>& network_input,
  const std::vector<std::vector<double>>& label_data
){
  RFASSERT_SCOPE(AUTODIFF_CALCULATE);
  for(std::uint32_t run_index = 0; run_index < network_input.size(); ++run_index){
    data.step();
    for(std::int32_t operation_index = operations.size() - 1; operation_index >= 0; --operation_index){
      operations[operation_index]->calculate_value(network_input[run_index], label_data[run_index]);
    }
    for(std::int32_t weight_index = 0u; weight_index < network.weight_table_size(); ++weight_index){
      for(std::int32_t operation_index = operations.size() - 1; operation_index >= 0; --operation_index)
        operations[operation_index]->calculate_derivative(
          static_cast<std::uint32_t>(weight_index), network_input[run_index], label_data[run_index]
        );
    }
  }
}

double RafkoAutodiffOptimizer::get_avg_gradient(std::uint32_t d_w_index){
  double sum = 0.0;
  double count = 0.0;
  for(std::uint32_t run_index = 0; run_index < network.memory_size(); ++run_index){
    for(std::uint32_t output_index = 0; output_index < network.output_neuron_number(); ++output_index){
      sum += data.get_derivative(run_index, output_index, d_w_index);
      count += 1.0;
    }
  }
  return sum / count;
}

#if(RAFKO_USES_OPENCL)
std::string RafkoAutodiffOptimizer::value_kernel_function(std::uint32_t output_index) const{
  std::uint32_t neuron_index = (network.neuron_array_size() - network.output_neuron_number() + output_index);
  auto found_element = neuron_spike_to_operation_map.find(neuron_index);
  RFASSERT(found_element != neuron_spike_to_operation_map.end());
  return operations[found_element->second]->value_kernel_function();
}
std::string RafkoAutodiffOptimizer::derivative_kernel_function() const{
  return "";
}
#endif/*(RAFKO_USES_OPENCL)*/

std::shared_ptr<RafkoBackpropagationOperation> RafkoAutodiffOptimizer::find_or_add_spike(std::uint32_t neuron_index){
  auto found_element = neuron_spike_to_operation_map.find(neuron_index);
  if(found_element != neuron_spike_to_operation_map.end())
    return operations[found_element->second];

  operations.emplace_back(new RafkoBackpropSpikeFnOperation(
    data, network, operations.size(), neuron_index
  ));
  neuron_spike_to_operation_map.insert( {neuron_index, (operations.size() - 1u)} );
  return operations.back();
}

std::shared_ptr<RafkoBackpropagationOperation> RafkoAutodiffOptimizer::push_dependency(DependencyParameter arguments){
  RFASSERT_LOG("Trying to push back operation {}", Autodiff_operations_Name(std::get<0>(arguments)));
  RFASSERT_LOGV(std::get<1>(arguments), "With parameters: ");
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
    case ad_operation_neuron_bias_d: //TODO: Have biases stored in a map, and re-used upon adding
                                      //this is possible based on weight index
      RFASSERT(2u == std::get<1>(arguments).size());
      return operations.emplace_back(new RafkoBackpropNeuronBiasOperation(
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

} /* namespace rafko_gym */
