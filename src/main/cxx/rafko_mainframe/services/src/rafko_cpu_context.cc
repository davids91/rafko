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

#include "rafko_mainframe/services/rafko_cpu_context.h"

#include <math.h>

#include "rafko_protocol/training.pb.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_dummies.h"

namespace rafko_mainframe{

RafkoCPUContext::RafkoCPUContext(rafko_net::RafkoNet& neural_network, rafko_mainframe::RafkoSettings settings_)
: settings(settings_.set_arena_ptr(&arena))
, network(neural_network)
, network_solution(rafko_net::SolutionBuilder(settings).build(network))
, agent(rafko_net::SolutionSolver::Builder(*network_solution, settings).build())
, environment(std::make_unique<RafkoDummyEnvironment>(network.input_data_size(), network.output_neuron_number()))
, objective(std::make_unique<RafkoDummyObjective>())
, weight_updater(rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, rafko_gym::weight_updater_amsgrad, settings))
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * environment->get_sequence_size() + 1u),
  std::vector<sdouble32>(network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
),execution_threads(settings.get_max_processing_threads())
{
  (void)settings.set_minibatch_size(std::max(1u,std::min(
    environment->get_number_of_sequences(),settings.get_minibatch_size()
  )));
  (void)settings.set_memory_truncation(std::max(1u,std::min(
    environment->get_sequence_size(), settings.get_memory_truncation()
  )));
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
}

void RafkoCPUContext::evaluate(uint32 sequence_start, uint32 sequences_to_evaluate, uint32 start_index_in_sequence, uint32 sequence_truncation){
  assert(environment->get_number_of_sequences() >= (sequence_start + sequences_to_evaluate));
  // std::unique_ptr<rafko_net::SolutionSolver> agent__ = rafko_net::SolutionSolver::Builder(*network_solution, settings).build();
  // rafko_net::SolutionSolver& agent_ = *agent__;
  objective->expose_to_multithreading();
  for(uint32 sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += settings.get_max_processing_threads()){
    execution_threads.start_and_block([this, sequence_index](uint32 thread_index){
      // std::unique_ptr<rafko_net::SolutionSolver> agent__ = (rafko_net::SolutionSolver::Builder(*network_solution, settings).build());

      if(environment->get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
        /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
         * Which is mainly because of division remainder between number fo threads and the number of sequences
         * */
        /* Solve the sequence under sequence_index + thread_index */
        uint32 raw_label_index = sequence_index + thread_index;
        uint32 raw_inputs_index = raw_label_index * (environment->get_sequence_size() + environment->get_prefill_inputs_number());
        raw_label_index *= environment->get_sequence_size();

        /* Evaluate the current sequence step by step */
        for(uint32 prefill_iterator = 0; prefill_iterator < environment->get_prefill_inputs_number(); ++prefill_iterator){
          // (void)agent_.solve(environment->get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
          (void)agent->solve(environment->get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
          ++raw_inputs_index;
        } /* The first few labels are there to set an initial state to the network */

        /* Solve the data and store the result after the inital "prefill" */
        for(uint32 sequence_iterator = 0; sequence_iterator < environment->get_sequence_size(); ++sequence_iterator){
          rafko_utilities::ConstVectorSubrange<> neuron_output = agent->solve(
            environment->get_input_sample(raw_inputs_index), ( (0u == environment->get_prefill_inputs_number())&&(0u == sequence_iterator) ), thread_index
          );
          std::copy( /* copy the result to the eval array */
            neuron_output.begin(), neuron_output.end(),
            neuron_outputs_to_evaluate[(thread_index * environment->get_sequence_size()) + sequence_iterator].begin()
          );
          ++raw_label_index;
          ++raw_inputs_index;
        }
      }
    });

    objective->set_features_for_sequences( /* Upload results to the data set */
      neuron_outputs_to_evaluate, 0u,
      sequence_index, std::min(((sequence_start + sequences_to_evaluate) - (sequence_index)), static_cast<uint32>(settings.get_max_processing_threads())),
      start_index_in_sequence, sequence_truncation, neuron_outputs_to_evaluate.back()
    );
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */
  objective->conceal_from_multithreading();
}

} /* namespace rafko_mainframe */
