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
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>

#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_utilities/models/subscript_proxy.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_backpropagation_data.h"

#include "rafko_gym/services/updater_factory.h"
#include "rafko_gym/services/rafko_backpropagation_operation.h"

namespace rafko_gym{

/**
 * @brief A class to calculate the values and derivatives of a Network, and update its weights based on it
 */
class RAFKO_FULL_EXPORT RafkoAutodiffOptimizer{
public:
  RafkoAutodiffOptimizer(const RafkoEnvironment& environment_, rafko_net::RafkoNet& network_, const rafko_mainframe::RafkoSettings& settings_)
  : settings(settings_)
  , environment(environment_)
  , network(network_)
  , data(network)
  , weight_updater(UpdaterFactory::build_weight_updater(network, weight_updater_default, settings))
  , neuron_spike_to_operation_map(std::make_shared<rafko_utilities::SubscriptDictionary>())
  , used_sequence_truncation( std::min(settings.get_memory_truncation(), environment.get_sequence_size()) )
  , used_minibatch_size( std::min(settings.get_minibatch_size(), environment.get_number_of_sequences()) )
  {
  }

  /**
   * @brief          Accepts a weight updater type to make handle the weight updates
   *
   * @param[in]      objective    An objective function ready to be moved inside the context
   */
  void set_weight_updater(rafko_gym::Weight_updaters updater){
    RFASSERT_LOG("Setting weight updater in Autodiff optimizer to {}", rafko_gym::Weight_updaters_Name(updater));
    weight_updater.reset();
    weight_updater = rafko_gym::UpdaterFactory::build_weight_updater(network, updater, settings);
  }

  /**
   * @brief   build or re-build the operateions based on the provided parameters
   *
   * @param[in]   objective     The objective function evaluating the network output
   */
  void build(const RafkoObjective& objective);

  /**
   * @brief     calculates the values and derivatives from the provided inputs and the stored Network reference
   *
   * @param[in]   network_input   The input the network is provided to produce its result
   * @param[in]   label_data      The values the network is expected to produce
   */
  using BackpropDataBufferRange = rafko_utilities::ConstVectorSubrange<std::vector<std::vector<double>>::const_iterator>;
  void calculate(BackpropDataBufferRange network_input, BackpropDataBufferRange label_data);

  /**
   * @brief   calculate the values and derivatives and update the weights based on them
   */
  void iterate();

  /**
   * @brief     provides a const reference to the calculated values of the network output
   *
   * @param[in]   past_index    The index of the run the network remembers
   *
   * @return      const reference of the data stored in the value buffers of the calculated network values
   */
  rafko_utilities::ConstVectorSubrange<> get_actual_value(std::uint32_t past_index){
    if(past_index > data.get_value().get_sequence_size())
      throw std::runtime_error("Reaching past value of Network beyond its memory");
    /*!Note: The first operations are for the Network output objectives, containing the output Neuron results */
    return {data.get_value().get_element(past_index).begin(), data.get_value().get_element(past_index).end()};
  }

  /**
   * @brief resets the internal buffers of the calculated values and derivatives
   */
  void reset(){
    data.reset();
  }

  /**
   * @brief Provides a const reference of the stored operations representing the objective comparison for a Neuron output
   *
   * @param[in]   neuron_index    the index of the Neuron to query
   *
   * @return    const access to one of the underlying outputoperations
   */
  const std::shared_ptr<RafkoBackpropagationOperation>& get_neuron_operation(std::uint32_t neuron_index){
    auto found_element = neuron_spike_to_operation_map->find(neuron_index);
    RFASSERT(found_element != neuron_spike_to_operation_map->end());
    return operations[found_element->second];
  }

  double get_avg_gradient(std::uint32_t d_w_index);

  #if(RAFKO_USES_OPENCL)
  std::string value_kernel_function(std::uint32_t output_index) const;
  std::string derivative_kernel_function() const;
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  const rafko_mainframe::RafkoSettings& settings;
  const RafkoEnvironment& environment;
  rafko_net::RafkoNet& network;
  RafkoBackpropagationData data;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;
  std::shared_ptr<rafko_utilities::SubscriptDictionary> neuron_spike_to_operation_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;

  const std::uint32_t used_sequence_truncation;
  const std::uint32_t used_minibatch_size;
  std::uint32_t iteration = 0u;

  void apply_weight_update(const std::vector<double>& weight_delta){
    RFASSERT_LOGV(weight_delta, "Applying weight(autodiff optimizer) update! Delta:");
    RFASSERT( static_cast<std::int32_t>(weight_delta.size()) == network.weight_table_size() );
    if(weight_updater->is_finished())
      weight_updater->start();
    weight_updater->iterate(weight_delta);
  }

  /**
   * @brief   calculate network value based on the given inputs
   *
   * @param[in]   network_input     the values the network takes as input
   */
  void calculate_value(const std::vector<double>& network_input);

  /**
   * @brief   calculate network derivative value for all weights based on the given inputs
   *
   * @param[in]   network_input     the values the network takes as input
   * @param[in]   label_data        the values the network output is compared to by the objective function
   */
  void calculate_derivative(
    const std::vector<double>& network_input, const std::vector<double>& label_data
  );

  std::shared_ptr<RafkoBackpropagationOperation> find_or_add_spike(std::uint32_t neuron_index);
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(DependencyParameter arguments);
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OPTIMIZER_H */
