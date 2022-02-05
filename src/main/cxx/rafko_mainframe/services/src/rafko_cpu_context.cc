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

RafkoCPUContext::RafkoCPUContext(rafko_net::RafkoNet neural_network_, rafko_mainframe::RafkoSettings settings_)
: settings(settings_)
, network(neural_network_)
, network_solution(rafko_net::SolutionBuilder(settings).build(network))
, agent(rafko_net::SolutionSolver::Builder(*network_solution, settings).build())
, environment(std::make_unique<RafkoDummyEnvironment>(network.input_data_size(), network.output_neuron_number()))
, objective(std::make_unique<RafkoDummyObjective>())
, weight_updater(rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, rafko_gym::weight_updater_amsgrad, settings))
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * environment->get_sequence_size() + 1u),
  std::vector<sdouble32>(network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
),execution_threads(settings.get_max_processing_threads())
, used_sequence_truncation( std::min(settings.get_memory_truncation(), environment->get_sequence_size()) )
, used_minibatch_size( std::min(settings.get_minibatch_size(), environment->get_number_of_sequences()) )
{ neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples()); }

void RafkoCPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_){
  assert(environment_->get_feature_size() == network.output_neuron_number());
  assert(environment_->get_input_size() == network.input_data_size());
  environment.reset();
  environment = environment_;
  uint32 old_output_buffer_num = neuron_outputs_to_evaluate.size();
  uint32 new_output_buffer_num = settings.get_max_processing_threads() * environment->get_sequence_size() + 1u;
  neuron_outputs_to_evaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(uint32 buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      neuron_outputs_to_evaluate[buffer_index].resize(environment->get_feature_size());
    }
  }
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  used_sequence_truncation = std::min(settings.get_memory_truncation(), environment->get_sequence_size());
  used_minibatch_size = std::min(settings.get_minibatch_size(), environment->get_number_of_sequences());
}

std::mutex cout_mutex;
sdouble32 RafkoCPUContext::evaluate(uint32 sequence_start, uint32 sequences_to_evaluate, uint32 start_index_in_sequence, uint32 sequence_truncation){
  assert(environment->get_number_of_sequences() >= (sequence_start + sequences_to_evaluate));

  sdouble32 error_sum = double_literal(0.0);
  // std::cout << "CPU run: \n";
  for(uint32 sequence_index = sequence_start; sequence_index < (sequence_start + sequences_to_evaluate); sequence_index += settings.get_max_processing_threads()){
    execution_threads.start_and_block([this, sequence_index](uint32 thread_index){
      if(environment->get_number_of_sequences() > (sequence_index + thread_index)){ /* See if the sequence index is inside bounds */
        /*!Note: This might happen because of the number of used threads might point to a grater index, than the number of sequences;
         * Which is mainly because of division remainder between number fo threads and the number of sequences
         * */
        /* Solve the sequence under sequence_index + thread_index */
        uint32 raw_inputs_index = (sequence_index + thread_index) * (environment->get_sequence_size() + environment->get_prefill_inputs_number());
        // std::lock_guard<std::mutex> lock(cout_mutex);

        /* Evaluate the current sequence step by step */
        for(uint32 prefill_iterator = 0; prefill_iterator < environment->get_prefill_inputs_number(); ++prefill_iterator){
          rafko_utilities::ConstVectorSubrange<> neuron_output = agent->solve(environment->get_input_sample(raw_inputs_index), (0 == prefill_iterator), thread_index);
          for(const sdouble32& n : neuron_output)std::cout << "[" << n <<"]";
          // std::cout << std::endl;
          ++raw_inputs_index;
        } /* The first few labels are there to set an initial state to the network */
        // std::cout << " -/- \n";
        /* Solve the data and store the result after the inital "prefill" */
        for(uint32 sequence_iterator = 0; sequence_iterator < environment->get_sequence_size(); ++sequence_iterator){
          rafko_utilities::ConstVectorSubrange<> neuron_output = agent->solve(
            environment->get_input_sample(raw_inputs_index), ( (0u == environment->get_prefill_inputs_number())&&(0u == sequence_iterator) ), thread_index
          );
          for(const sdouble32& n : neuron_output)std::cout << "[" << n <<"]";
          // std::cout << std::endl;
          std::copy( /* copy the result to the eval array */
            neuron_output.begin(), neuron_output.end(),
            neuron_outputs_to_evaluate[(thread_index * environment->get_sequence_size()) + sequence_iterator].begin()
          );
          ++raw_inputs_index;
        }
      }
    });
    /*!Debug: see what is evaluated */
    // std::cout << "CPU Evaluation:\n";
    // uint32 num_to_eval = environment->get_sequence_size() * std::min(
    //   ((sequence_start + sequences_to_evaluate) - (sequence_index)),
    //   static_cast<uint32>(settings.get_max_processing_threads())
    // );
    // uint32 num_evaled = 0;
    // uint32 raw_label_start = (sequence_index * environment->get_sequence_size());
    // uint32 raw_input_start =
    //   sequence_index * (environment->get_sequence_size() + environment->get_prefill_inputs_number());
    // uint32 input_offset = 0;
    // for(const std::vector<sdouble32>& feature : neuron_outputs_to_evaluate){
    //   if(num_evaled < num_to_eval){
    //     if(0 == (input_offset%(environment->get_sequence_size() + environment->get_prefill_inputs_number())))
    //       input_offset += environment->get_prefill_inputs_number();
    //     std::cout << "<";
    //     for(const sdouble32& input : environment->get_input_sample(raw_input_start + input_offset))
    //       std::cout << "[" << input << "]";
    //     std::cout << ">";
    //     for(const sdouble32& feature_ : feature){
    //       std::cout << "[" << feature_ << "]";
    //       std::cout << "<>[(" << (raw_label_start + num_evaled) << ")" << environment->get_label_sample(raw_label_start + num_evaled)[0] << "]";
    //     }
    //     ++input_offset;
    //     std::cout << std::endl;
    //   }
    //   ++num_evaled;
    // }
    sdouble32 error_part = objective->set_features_for_sequences( /* Upload results to the data set */
      *environment, neuron_outputs_to_evaluate,
      0u/* neuron_buffer_index */, sequence_index, std::min(
        ((sequence_start + sequences_to_evaluate) - (sequence_index)),
        static_cast<uint32>(settings.get_max_processing_threads())
      )/* sequences_to_evaluate */,
      start_index_in_sequence, sequence_truncation, neuron_outputs_to_evaluate.back()
    );
    // std::cout << "error_part: " << error_part << std::endl;
    // std::cout << "============" << std::endl;
    error_sum += error_part;
  } /* for(sequence_index: sequence_start --> (sequence start + sequences_to_evaluate)) */
  // std::cout << "========================" << std::endl;
  return -( error_sum / static_cast<sdouble32>(sequences_to_evaluate * environment->get_sequence_size()) );
}

} /* namespace rafko_mainframe */
