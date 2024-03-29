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

#include "rafko_global.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rafko_gym/models/rafko_backpropagation_data.hpp"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_mainframe/models/rafko_autonomous_entity.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_mainframe/services/rafko_context.hpp"
#include "rafko_utilities/models/const_vector_subrange.hpp"
#include "rafko_utilities/models/subscript_proxy.hpp"

#include "rafko_gym/services/rafko_backprop_spike_fn_operation.hpp"
#include "rafko_gym/services/rafko_backpropagation_operation.hpp"
#include "rafko_gym/services/updater_factory.hpp"

namespace rafko_gym {

/**
 * @brief A class to calculate the values and derivatives of a Network, and
 * update its weights based on it
 */
class RAFKO_EXPORT RafkoAutodiffOptimizer
    : public rafko_mainframe::RafkoAutonomousEntity {
  using BackpropDataBufferRange = rafko_utilities::ConstVectorSubrange<
      std::vector<std::vector<double>>::const_iterator>;

public:
  RafkoAutodiffOptimizer(
      std::shared_ptr<rafko_mainframe::RafkoSettings> settings,
      rafko_net::RafkoNet &network,
      std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator = {},
      std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator = {})
      : rafko_mainframe::RafkoAutonomousEntity(settings), m_network(network),
        m_data(network), m_weightUpdater(UpdaterFactory::build_weight_updater(
                             m_network, weight_updater_default, *m_settings)),
        m_neuronIndexToSpikeOperationIndex(m_network.neuron_array_size()),
        m_executionThreads(), m_trainingEvaluator(training_evaluator),
        m_testEvaluator(test_evaluator),
        m_tmpAvgD(m_network.weight_table_size()) {
    for (std::uint32_t thread_index = 0;
         thread_index < m_settings->get_max_processing_threads();
         ++thread_index)
      m_executionThreads.push_back(
          std::make_unique<rafko_utilities::ThreadGroup>(
              m_settings->get_max_solve_threads()));
  }

  /**
   * @brief     Provides information on when to stop the training according to
   * the strategies provided in the settings
   *
   * @return    True, if training can be stopped
   */
  bool stop_triggered() const {
    return ((/* Early stopping */
             (m_trainingEvaluator && m_testEvaluator) &&
             ((m_settings->get_training_strategy(
                  Training_strategy::training_strategy_early_stopping)) &&
              (m_lastTrainingError >
               (m_lastTestingError * (1.0 + m_settings->get_delta()))))) ||
            ((m_settings->get_training_strategy(
                 Training_strategy::
                     training_strategy_stop_if_training_error_zero)) &&
             (0.0 == m_lastTrainingError)));
  }

  /**
   * @brief          Accepts a weight updater type to handle the weight updates
   *
   * @param[in]      objective    An objective function ready to be moved inside
   * the context
   */
  void set_weight_updater(rafko_gym::Weight_updaters updater) {
    RFASSERT_LOG("Setting weight updater in Autodiff optimizer to {}",
                 rafko_gym::Weight_updaters_Name(updater));
    m_weightUpdater.reset();
    m_weightUpdater = rafko_gym::UpdaterFactory::build_weight_updater(
        m_network, updater, *m_settings);
  }

  /**
   * @brief          Accepts a RafkoContext shared pointer to use to produce the
   * training error values
   *
   * @param[in]      context    A RafkoContext shared pointer
   */
  void
  set_training_context(std::shared_ptr<rafko_mainframe::RafkoContext> context) {
    m_trainingEvaluator = context;
  }

  /**
   * @brief          Accepts a RafkoContext shared pointer to use to produce the
   * testing error values
   *
   * @param[in]      context    A RafkoContext shared pointer
   */
  void
  set_testing_context(std::shared_ptr<rafko_mainframe::RafkoContext> context) {
    m_testEvaluator = context;
  }

  /**
   * @brief   build or re-build the operateions based on the provided parameters
   *
   * @param[in]   data_set      The data set the network is evaluated on,
   * required to set buffer sizes and to add needed parameters for objective
   * operations
   * @param       objective     The objective function evaluating the network
   * output
   */
  virtual void build(const std::shared_ptr<RafkoDataSet> data_set,
                     std::shared_ptr<RafkoObjective> objective = {});

  /**
   * @brief     calculates the values and derivatives from the provided inputs
   * and the stored Network reference
   *
   * @param[in]   network_input   The input the network is provided to produce
   * its result
   * @param[in]   label_data      The values the network is expected to produce
   */
  void calculate(BackpropDataBufferRange network_input,
                 BackpropDataBufferRange label_data);

  /**
   * @brief   calculate the values and derivatives and update the weights based
   * on them
   *
   * @param[in]   data_set            The data set the network is evaluated on
   * @param[in]   force_gpu_upload    Force upload inpuat and label data to GPU,
   * should it be relevant in used test/training contexts
   */
  virtual void iterate(const RafkoDataSet &data_set,
                       bool force_gpu_upload = false);

  /**
   * @brief     provides a const reference to the calculated values of the
   * network output
   *
   * @param[in]   past_index    The index of the run the network remembers
   *
   * @return      const reference of the data stored in the value buffers of the
   * calculated network values
   */
  rafko_utilities::ConstVectorSubrange<>
  get_actual_value(std::uint32_t past_index) {
    if (past_index > m_data.get_value().get_sequence_size())
      throw std::runtime_error(
          "Reaching past value of Network beyond its memory");
    /*!Note: The first operations are for the Network output objectives,
     * containing the output Neuron results */
    return {m_data.get_value().get_element(past_index).begin(),
            m_data.get_value().get_element(past_index).end()};
  }

  /**
   * @brief resets the internal buffers of the calculated values and derivatives
   */
  void reset() { m_data.reset(); }

  /**
   * @brief     Provides a const reference of the stored operations representing
   * the objective comparison for a Neuron output
   *
   * @param[in]   neuron_index    the index of the Neuron to query
   *
   * @return    const access to one of the underlying output operations
   */
  const std::shared_ptr<RafkoBackpropagationOperation> &
  get_neuron_operation(std::uint32_t neuron_index) const {
    return m_operations[get_operation_index(neuron_index)];
  }

  /**
   * @brief     Provides the activation value of a single neuron
   *
   * @param[in]   past_index        The past index to take the value from
   * @param[in]   neuron_index      The relevant neuron index
   */
  double get_neuron_data(std::uint32_t past_index, std::uint32_t neuron_index) {
    return get_neuron_operation(neuron_index)->get_value(past_index);
  }

  /**
   * @brief     Calcualtes the average gradient for one weight from the last
   * iteration
   *
   * @param[in]   d_w_index     the index of the weight to get the average
   * gradient for
   *
   * @return    the average gradient for the weight under the given index
   */
  virtual double get_avg_gradient(std::uint32_t d_w_index) const;

  /**
   * @brief   Calculates the average of the absolute value of the
   *          gradient for every weight, providing a blurry insight onto the
   * training surface
   *
   * @return    the average of the weight gradient values
   */
  double get_avg_of_abs_gradient() const {
    double sum = 0.0;
    for (std::int32_t weight_index = 0;
         weight_index < m_network.weight_table_size(); ++weight_index) {
      sum += std::abs(get_avg_gradient(weight_index));
    }
    return sum / static_cast<double>(m_network.weight_table_size());
  }

  /**
   * @brief     provides the last measured training error from the testing
   * evaluator provided at cunstruction
   *
   * @return    the error value
   */
  constexpr double get_last_training_error() const {
    return m_lastTrainingError;
  }

  /**
   * @brief     provides the last measured testing error from the testing
   * evaluator provided at cunstruction
   *
   * @return    the error value
   */
  constexpr double get_last_testing_error() const { return m_lastTestingError; }

  /**
   * @brief     Provides information about the optimizer built state
   *
   * @return    True, if the optimizer is built correctly
   */
  constexpr bool built() { return m_built; }

  /**
   * @brief     Resets the iteration count to zero
   */
  void reset_epoch() { m_iteration = 1; }

protected:
  rafko_net::RafkoNet &m_network;
  RafkoBackpropagationData m_data;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> m_weightUpdater;
  static constexpr std::uint32_t s_NeuronNotYetAssigned = static_cast<std::uint32_t>(-1);
  std::vector<std::uint32_t> m_neuronIndexToSpikeOperationIndex;
  std::unordered_map<std::uint32_t,
                     std::shared_ptr<RafkoBackpropSpikeFnOperation>>
      m_unplacedSpikes;
  std::unordered_map<std::uint32_t, std::uint32_t> m_spikeSolvesFeatureMap;
  std::vector<std::shared_ptr<RafkoBackpropagationOperation>> m_operations;
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> m_executionThreads;

  std::shared_ptr<rafko_mainframe::RafkoContext> m_trainingEvaluator;
  std::shared_ptr<rafko_mainframe::RafkoContext> m_testEvaluator;

  std::uint32_t m_usedSequenceTruncation = 0;
  std::uint32_t m_usedMinibatchSize = 0;
  std::uint32_t m_iteration = 0u;
  std::uint32_t m_lastTestedIteration = 0u;
  double m_lastTrainingError = std::numeric_limits<double>::quiet_NaN();
  double m_lastTestingError = std::numeric_limits<double>::quiet_NaN();
  bool m_built = false;
  std::vector<double> m_tmpAvgD;

  /**
   * @brief     Queries the index of the output operation of the given neuron
   * index
   *
   * @param[in]   neuron_index    the index of the Neuron to query
   *
   * @return    the operation index
   */
  std::uint32_t get_operation_index(std::uint32_t neuron_index) const {
    RFASSERT(neuron_index < m_neuronIndexToSpikeOperationIndex.size());
    return m_neuronIndexToSpikeOperationIndex[neuron_index];
  }

  /**
   * @brief   Calculates the training and test set values where appropriate
   *
   * @param[in]      force_gpu_upload     If set true, data in stored objects
   * are uploaded to GPU regardless of previous uploads
   */
  void update_context_errors(bool force_gpu_upload);

  /**
   * @brief   applies a weight update to the network
   *
   * @param[in]   weight_delta    the weight gradients to update it with
   */
  void apply_weight_update(const std::vector<double> &weight_delta) {
    RFASSERT_LOGV(weight_delta,
                  "Applying weight(autodiff optimizer) update! Delta:");
    RFASSERT(static_cast<std::int32_t>(weight_delta.size()) ==
             m_network.weight_table_size());
    if (m_weightUpdater->is_finished())
      m_weightUpdater->start();
    m_weightUpdater->iterate(weight_delta);
  }

  /**
   * @brief   build or re-build the operateions based on the provided parameters
   *
   * @param[in]   data_set      The data set the network is evaluated on
   * @param       objective     The objective function evaluating the network
   * output
   *
   * @return  The number of operations at the start of the array directly
   * relevant to weight derivatives
   */
  std::uint32_t build_without_data(const std::shared_ptr<RafkoDataSet> data_set,
                                   std::shared_ptr<RafkoObjective> objective);

  /**
   * @brief   calculate network value based on the given inputs
   *
   * @param[in]   network_input     the values the network takes as input
   */
  void calculate_value(const std::vector<double> &network_input);

  /**
   * @brief   calculate network derivative value for all weights based on the
   * given inputs
   *
   * @param[in]   network_input     the values the network takes as input
   * @param[in]   label_data        the values the network output is compared to
   * by the objective function
   */
  void calculate_derivative(const std::vector<double> &network_input,
                            const std::vector<double> &label_data);

  /**
   * @brief   Inserts the spike function operation to teh operations map for the
   * given Neuron index; Looks into the unplaced map first, if there's already
   * an unplaced spike it inserts it from there
   *
   * @param[in]   neuron_index    the index of the Neuron Spike to place
   * @param[in]   dependencies    An array of dependencies the Neuron operation
   * might have
   *
   * @return    A shared pointer of the placed operation
   */
  std::shared_ptr<RafkoBackpropagationOperation> place_spike_to_operations(
      std::uint32_t neuron_index,
      std::vector<RafkoBackpropagationOperation::Dependency> dependencies = {});

  /**
   * @brief   Inserts the spike function operation into the unplaced map;
   *          Or finds the index in it and returns with the pointer to it
   *
   * @param[in]   neuron_index    the index of the Neuron Spike to place
   *
   * @return    A shared pointer of the operation
   */
  std::shared_ptr<RafkoBackpropagationOperation>
  find_or_queue_spike(std::uint32_t neuron_index);

  /**
   * @brief   Places the dependency either into the operations array, or maybe
   * the unplaced map;
   *
   * @param[in]   DependencyParameter   the parameters of the operation to place
   *
   * @return    A shared pointer of the placed operation
   */
  std::shared_ptr<RafkoBackpropagationOperation>
  push_dependency(RafkoBackpropagationOperation::DependencyParameter arguments);
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OPTIMIZER_H */
