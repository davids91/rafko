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

#include "rafko_mainframe/services/rafko_gpu_context.h"

#include <assert.h>

#include "rafko_net/services/solution_builder.h"
#include "rafko_mainframe/services/rafko_dummies.h"


namespace rafko_mainframe{

RafkoGPUContext::Builder::Builder(rafko_net::RafkoNet neural_network_, rafko_mainframe::RafkoSettings settings_)
: settings(settings_)
, network(neural_network_)
{
  cl::Platform::get(&platforms);
  assert( 0 < platforms.size() );
}

RafkoGPUContext::Builder& RafkoGPUContext::Builder::select_platform(uint32 platform_index){
  assert( platform_index < platforms.size() );
  selected_platform = platform_index;
  return *this;
}

RafkoGPUContext::Builder& RafkoGPUContext::Builder::select_device(cl_device_type type, uint32 device_index){
  platforms[selected_platform].getDevices(type, &devices);
  assert( device_index < devices.size() );
  selected_device = device_index;
  return *this;
}

std::unique_ptr<RafkoGPUContext> RafkoGPUContext::Builder::build(){
  assert( 0 < platforms.size() );
  assert( 0 < devices.size() );
  cl::Context context({devices[selected_device]});
  return std::unique_ptr<RafkoGPUContext>(
    new RafkoGPUContext(std::move(context), std::move(settings), std::move(network))
  );
}

RafkoGPUContext::RafkoGPUContext(
  cl::Context&& context_, rafko_mainframe::RafkoSettings&& settings_, rafko_net::RafkoNet&& neural_network_
):settings(settings_.set_arena_ptr(&arena))
, opencl_context(context_)
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

void RafkoGPUContext::set_network_weight(uint32 weight_index, sdouble32 weight_value){
  assert( static_cast<sint32>(weight_index) < network.weight_table_size() );
  network.set_weight_table(weight_index, weight_value);
  weight_updater->update_solution_with_weights();
  #warning "Need to update weight in GPU buffers as well"
}

void RafkoGPUContext::set_network_weights(const std::vector<sdouble32>& weights){
  assert( static_cast<sint32>(weights.size()) == network.weight_table_size() );
  *network.mutable_weight_table() = {weights.begin(), weights.end()};
  weight_updater->update_solution_with_weights();
  #warning "Need to update weights in GPU buffers as well"
}

void RafkoGPUContext::apply_weight_update(const std::vector<sdouble32>& weight_delta){
  assert( static_cast<sint32>(weight_delta.size()) == network.weight_table_size() );
  if(weight_updater->is_finished())
    weight_updater->start();
  weight_updater->iterate(weight_delta);
  weight_updater->update_solution_with_weights();
  #warning "Need to update weights in GPU buffers as well"
}

void RafkoGPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_){
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

sdouble32 RafkoGPUContext::full_evaluation(){
#warning "full_evaluation not implemented yet"
}

sdouble32 RafkoGPUContext::stochastic_evaluation(bool to_seed, uint32 seed_value){
  #warning "stochastic_evaluation not implemented yet"
}

rafko_utilities::ConstVectorSubrange<> RafkoGPUContext::solve(
  const std::vector<sdouble32>& input,
  bool reset_neuron_data, uint32 thread_index
){
  #warning "solve not implemented yet"

}

} /* namespace rafko_mainframe */
