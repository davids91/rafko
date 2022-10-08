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

#ifndef RAFKO_CPU_CONTEXT_H
#define RAFKO_CPU_CONTEXT_H

#include "rafko_global.hpp"

#include <memory>

#include "rafko_net/services/solution_solver.hpp"
#include "rafko_gym/models/rafko_environment.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/services/updater_factory.hpp"

#include "rafko_mainframe/services/rafko_context.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_mainframe {

class RAFKO_EXPORT RafkoCPUContext : public RafkoContext{
public:

  RafkoCPUContext(
    rafko_net::RafkoNet& neural_network,
    std::shared_ptr<rafko_mainframe::RafkoSettings> = {},
    std::shared_ptr<rafko_gym::RafkoObjective> objective = {}
  );
  ~RafkoCPUContext() = default;

  /* +++ Methods taken from @RafkoContext +++ */
  void set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment) override;

  void set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective) override{
    RFASSERT_LOG("Setting objective in CPU Context");
    m_objective = objective;
  }

  void set_weight_updater(rafko_gym::Weight_updaters updater) override{
    RFASSERT_LOG("Setting weight updater in CPU context to {}", rafko_gym::Weight_updaters_Name(updater));
    m_weightUpdater.reset();
    m_weightUpdater = rafko_gym::UpdaterFactory::build_weight_updater(m_network, updater, *m_settings);
  }

  void refresh_solution_weights() override{
    RFASSERT_LOG("Refreshing Solution weights in CPU context..");
    m_solverFactory.refresh_actual_solution_weights();
  }

  void set_network_weight(std::uint32_t weight_index, double weight_value) override{
    RFASSERT_LOG("Setting weight[{}] to {}(CPU Context)", weight_index, weight_value);
    RFASSERT( static_cast<std::int32_t>(weight_index) < m_network.weight_table_size() );
    m_network.set_weight_table(weight_index, weight_value);
    refresh_solution_weights();
  }

  void set_network_weights(const std::vector<double>& weights) override{
    RFASSERT_LOGV(weights, "Setting weights(CPU Context) to:");
    RFASSERT( static_cast<std::int32_t>(weights.size()) == m_network.weight_table_size() );
    *m_network.mutable_weight_table() = {weights.begin(), weights.end()};
    refresh_solution_weights();
  }

  void apply_weight_update(const std::vector<double>& weight_delta) override{
    RFASSERT_LOGV(weight_delta, "Applying weight(CPU context) update! Delta:");
    RFASSERT( static_cast<std::int32_t>(weight_delta.size()) == m_network.weight_table_size() );
    if(m_weightUpdater->is_finished())
      m_weightUpdater->start();
    m_weightUpdater->iterate(weight_delta);
    refresh_solution_weights();
  }

  double full_evaluation() override{
    RFASSERT_SCOPE(CPU_FULL_EVALUATION);
    return evaluate(
      0u, m_environment->get_number_of_sequences(),
      0u, m_environment->get_sequence_size()
    );
  }

  double stochastic_evaluation(bool to_seed = false, std::uint32_t seed_value = 0u) override{
    RFASSERT_SCOPE(CPU_STOCHASTIC_EVALUATION);
    if(to_seed)srand(seed_value);
    std::uint32_t sequence_start_index = (rand()%(m_environment->get_number_of_sequences() - m_usedMinibatchSize + 1));
    std::uint32_t start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
      m_environment->get_sequence_size() - m_usedSequenceTruncation + 1u /* ..not all result output values are evaluated.. */
    )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */
    return evaluate(
      sequence_start_index, m_usedMinibatchSize,
      start_index_inside_sequence, m_usedSequenceTruncation
    );
  }

  rafko_utilities::ConstVectorSubrange<> solve(
    const std::vector<double>& input,
    bool reset_neuron_data = false, std::uint32_t thread_index = 0
  ) override{
    RFASSERT_SCOPE(CPU_STANDALONE_SOLVE);
    return m_agent->solve(input, reset_neuron_data, thread_index);
  }

  void solve_environment(std::vector<std::vector<double>>& output, bool isolated = true) override;

  void push_state() override{
    m_environment->push_state();
  }

  void pop_state() override{
    m_environment->pop_state();
  }

  rafko_mainframe::RafkoSettings& expose_settings() override{
    return *m_settings;
  }

  constexpr rafko_net::RafkoNet& expose_network() override{
    return m_network;
  }
  /* --- Methods taken from @RafkoContext --- */

private:
  rafko_net::RafkoNet& m_network;
  rafko_net::SolutionSolver::Factory m_solverFactory;
  std::shared_ptr<rafko_net::SolutionSolver> m_agent;
  std::shared_ptr<rafko_gym::RafkoEnvironment> m_environment;
  std::shared_ptr<rafko_gym::RafkoObjective> m_objective;
  std::shared_ptr<rafko_gym::RafkoWeightUpdater> m_weightUpdater;

  std::vector<std::vector<double>> m_neuronOutputsToEvaluate; /* for each feature array inside each sequence inside each thread in one evaluation iteration */
  rafko_utilities::ThreadGroup m_executionThreads;

  std::uint32_t m_usedSequenceTruncation;
  std::uint32_t m_usedMinibatchSize;

  /**
   * @brief      Evaluate the given data set with the given parameters
   *
   * @param[in]  sequence_start             The starting sequence to be evaluated inside the @data_set
   * @param[in]  sequences_to_evaluate      The number of sequences to evaluate inside the @data_set
   * @param[in]  start_index_in_sequence    Parameter for sequence truncation: only update error value starting from this index in every sequence
   * @param[in]  sequence_tructaion         The number of labels to evaluate inside every evaluated sequence
   * @return     The resulting fitness
   */
  double evaluate(std::uint32_t sequence_start, std::uint32_t sequences_to_evaluate, std::uint32_t start_index_in_sequence, std::uint32_t sequence_tructaion);


  double error_post_process(double raw_error, std::uint32_t labels_evaluated);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_CPU_CONTEXT_H */
