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

#ifndef RAFKO_AUTODIFF_OPTIMIZER_H
#define RAFKO_AUTODIFF_OPTIMIZER_H

#include "rafko_global.h"

#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <limits>
#include <stdexcept>

#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_backpropagation_data.h"

#include "rafko_gym/services/rafko_backpropagation_operation.h"
#include "rafko_gym/services/rafko_backprop_network_input_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.h"
#include "rafko_gym/services/rafko_backprop_transfer_fn_operation.h"
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.h"
#include "rafko_gym/services/rafko_backprop_objective_operation.h"

namespace rafko_gym{

/**
 * @brief
 *
 */
class RAFKO_FULL_EXPORT RafkoAutodiffOptimizer{
public:
  RafkoAutodiffOptimizer(const rafko_net::RafkoNet& network_, rafko_mainframe::RafkoSettings& settings_)
  : settings(settings_)
  , network(network_)
  , data(network)
  {
  }

  void build(RafkoEnvironment& environment, RafkoObjective& objective);

  void calculate(
    const std::vector<std::vector<double>>& network_input,
    const std::vector<std::vector<double>>& label_data
  );

  rafko_utilities::ConstVectorSubrange<> get_actual_value(std::uint32_t past_index){
    if(past_index > data.get_value().get_sequence_size())
      throw std::runtime_error("Reaching past value of Network beyond its memory");
    return {data.get_value().get_element(past_index).begin(), data.get_value().get_element(past_index).end()};
  }

  void reset(){
    data.reset();
  }

  std::shared_ptr<RafkoBackpropagationOperation>& get_neuron_operation(std::uint32_t output_index){
    std::uint32_t neuron_index = (network.neuron_array_size() - network.output_neuron_number() + output_index);
    auto found_element = neuron_spike_to_operation_map.find(neuron_index);
    RFASSERT(found_element != neuron_spike_to_operation_map.end());
    return operations[found_element->second];
  }

  double get_avg_gradient(std::uint32_t d_w_index);

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function(std::uint32_t output_index) const;
  std::string derivative_kernel_function() const;
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  rafko_mainframe::RafkoSettings& settings;
  const rafko_net::RafkoNet& network;
  RafkoBackpropagationData data;
  std::map<std::uint32_t, std::uint32_t> neuron_spike_to_operation_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;

  std::shared_ptr<RafkoBackpropagationOperation> find_or_add_spike(std::uint32_t neuron_index);
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(DependencyParameter arguments);
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OPTIMIZER_H */
