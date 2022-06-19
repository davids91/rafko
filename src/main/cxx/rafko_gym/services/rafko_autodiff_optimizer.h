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
#include <unordered_map>
#include <limits>
#include <stdexcept>

#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_utilities/models/subscript_proxy.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_mainframe/services/rafko_context.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_backpropagation_data.h"

#include "rafko_gym/services/updater_factory.h"
#include "rafko_gym/services/rafko_backpropagation_operation.h"
#include "rafko_gym/services/rafko_backprop_spike_fn_operation.h"

namespace rafko_gym{

/**
 * @brief A class to calculate the values and derivatives of a Network, and update its weights based on it
 */
class RAFKO_FULL_EXPORT RafkoAutodiffOptimizer{
  using BackpropDataBufferRange = rafko_utilities::ConstVectorSubrange<std::vector<std::vector<double>>::const_iterator>;
public:
  RafkoAutodiffOptimizer(
    const rafko_mainframe::RafkoSettings& settings_,
    std::shared_ptr<RafkoEnvironment> environment_, rafko_net::RafkoNet& network_,
    std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator_ = {},
    std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator_ = {}
  )
  : settings(settings_)
  , environment(environment_)
  , network(network_)
  , data(network)
  , weight_updater(UpdaterFactory::build_weight_updater(network, weight_updater_default, settings))
  , neuron_spike_to_operation_map(std::make_shared<rafko_utilities::SubscriptDictionary>())
  , execution_threads()
  , training_evaluator(training_evaluator_)
  , test_evaluator(test_evaluator_)
  , used_sequence_truncation( std::min(settings.get_memory_truncation(), environment->get_sequence_size()) )
  , used_minibatch_size( std::min(settings.get_minibatch_size(), environment->get_number_of_sequences()) )
  , tmp_avg_derivatives(network.weight_table_size())
  {
    if(training_evaluator){
      training_evaluator->set_environment(environment);
    }
    for(std::uint32_t thread_index = 0; thread_index < settings.get_max_processing_threads(); ++ thread_index)
      execution_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(
        settings.get_max_solve_threads()
      ));
  }

  bool early_stopping_triggered(){
    return (
      (training_evaluator && test_evaluator)
      &&( last_training_error > ( last_testing_error * (1.0 + settings.get_delta()) ) )
    );
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
   * @param   objective     The objective function evaluating the network output
   */
  void build(std::shared_ptr<RafkoObjective> objective){
    std::uint32_t weight_relevant_operation_count = build_without_data(objective);
    data.build(
      operations.size(), weight_relevant_operation_count,
      environment->get_sequence_size()
    );
  }

  /**
   * @brief     calculates the values and derivatives from the provided inputs and the stored Network reference
   *
   * @param[in]   network_input   The input the network is provided to produce its result
   * @param[in]   label_data      The values the network is expected to produce
   */
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
   * @brief     Provides a const reference of the stored operations representing the objective comparison for a Neuron output
   *
   * @param[in]   neuron_index    the index of the Neuron to query
   *
   * @return    const access to one of the underlying output operations
   */
  const std::shared_ptr<RafkoBackpropagationOperation>& get_neuron_operation(std::uint32_t neuron_index) const{
    return operations[get_operation_index(neuron_index)];
  }

  double get_avg_gradient(std::uint32_t d_w_index);
  double get_avg_gradient(){
    double sum = 0.0;
    for(std::int32_t weight_index = 0; weight_index < network.weight_table_size(); ++weight_index){
      sum += get_avg_gradient(weight_index);
    }
    return sum / static_cast<double>(network.weight_table_size());
  }

  constexpr double get_last_training_error() const{
      return last_training_error;
  }

  constexpr double get_last_testing_error() const{
      return last_testing_error;
  }

protected:
  const rafko_mainframe::RafkoSettings& settings;
  std::shared_ptr<RafkoEnvironment> environment;
  rafko_net::RafkoNet& network;
  RafkoBackpropagationData data;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> weight_updater;
  std::shared_ptr<rafko_utilities::SubscriptDictionary> neuron_spike_to_operation_map;
  std::unordered_map<std::uint32_t, std::shared_ptr<RafkoBackpropSpikeFnOperation>> unplaced_spikes;
  std::unordered_map<std::uint32_t, std::uint32_t> spike_solves_feature_map;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> operations;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> execution_threads;

  std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator;
  std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator;

  const std::uint32_t used_sequence_truncation;
  const std::uint32_t used_minibatch_size;
  std::uint32_t iteration = 0u;
  double last_training_error = std::numeric_limits<double>::quiet_NaN();
  double last_testing_error = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> tmp_avg_derivatives;

  /**
   * @brief     Queries the index of the output operation of the given neuron index
   *
   * @param[in]   neuron_index    the index of the Neuron to query
   *
   * @return    the operation index
   */
  std::uint32_t get_operation_index(std::uint32_t neuron_index) const{
    auto found_element = neuron_spike_to_operation_map->find(neuron_index);
    RFASSERT(found_element != neuron_spike_to_operation_map->end());
    return found_element->second;
  }

  /**
   * @brief   Calculates the training and test set values where appropriate
   */
  void update_context_errors();

  /**
   * @brief   applies a weight update to the network
   *
   * @param[in]   weight_delta    the weight delta to update
   */
  void apply_weight_update(const std::vector<double>& weight_delta){
    RFASSERT_LOGV(weight_delta, "Applying weight(autodiff optimizer) update! Delta:");
    RFASSERT( static_cast<std::int32_t>(weight_delta.size()) == network.weight_table_size() );
    if(weight_updater->is_finished())
      weight_updater->start();
    weight_updater->iterate(weight_delta);
  }

  /**
   * @brief   build or re-build the operateions based on the provided parameters
   *
   * @param   objective     The objective function evaluating the network output
   *
   * @return  The number of operations at the start of the array directly relevant to weight derivatives
   */
  std::uint32_t build_without_data(std::shared_ptr<RafkoObjective> objective);


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

  /**
   * @brief   Inserts the spike function operation to teh operations map for the given Neuron index;
   *          Looks into the unplaced map first, if there's already an unplaced spike
   *          it inserts it from there
   *
   * @param[in]   neuron_index    the index of the Neuron Spike to place
   *
   * @return    A shared pointer of the placed operation
   */
  std::shared_ptr<RafkoBackpropagationOperation> place_spike_to_operations(std::uint32_t neuron_index);

  /**
   * @brief   Inserts the spike function operation into the unplaced map;
   *          Or finds the index in it and returns with the pointer to it
   *
   * @param[in]   neuron_index    the index of the Neuron Spike to place
   *
   * @return    A shared pointer of the operation
   */
  std::shared_ptr<RafkoBackpropagationOperation> find_or_queue_spike(std::uint32_t neuron_index);

  /**
   * @brief   Places the dependency either into the operations array, or maybe the unplaced map;
   *
   * @param[in]   DependencyParameter   the parameters of the operation to place
   *
   * @return    A shared pointer of the placed operation
   */
  std::shared_ptr<RafkoBackpropagationOperation> push_dependency(DependencyParameter arguments);
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OPTIMIZER_H */
