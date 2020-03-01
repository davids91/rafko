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

#include "services/sparse_net_optimizer.h"
#include "models/spike_function.h"
#include "services/neuron_router.h"

#include <atomic>
#include <thread>
#include <cmath>

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;

void Sparse_net_optimizer::step(uint32 mini_batch_size){
  /* Calculate features and errors for every input */
  for(unique_ptr<atomic<sdouble32>>& weight_gradient : weight_gradients) weight_gradient->store(0);
  for(uint32 thread_index = 0; thread_index < context.get_max_solve_threads();++thread_index){ /* For every provided sample */
    solve_threads.push_back(thread(&Sparse_net_optimizer::step_thread, this, thread_index, mini_batch_size));
  }
  wait_for_threads(solve_threads);

  /* Update the weights of the SparseNet and the solution */
  weight_updater->update_weights_with_gradients();
  weight_updater->update_solution_with_weights(net_solution);
}

void Sparse_net_optimizer::step_thread(uint32 solve_thread_index, uint32 samples_to_evaluate){
  uint32 sample_index;
  for(uint32 sample = 0; sample < samples_to_evaluate; ++sample){
    sample_index = rand()%(data_set.get_number_of_samples());
    solver[solve_thread_index].solve(data_set.get_input_sample(sample_index)); /* Solve the network for the given input */
    neuron_data[solve_thread_index] = solver[solve_thread_index].get_neuron_data();
    transfer_function_input[solve_thread_index] = solver[solve_thread_index].get_transfer_function_input();
    transfer_function_output[solve_thread_index] = solver[solve_thread_index].get_transfer_function_output();

    if(data_set.get_feature_size() != solver[solve_thread_index].get_output_size())
      throw "Network output size doesn't match size of provided labels!";

    data_set.set_feature_for_label(sample_index, neuron_data[solve_thread_index]);

    for(unique_ptr<atomic<sdouble32>>& error_value : error_values[solve_thread_index]) *error_value = 0;
    calculate_output_errors(solve_thread_index, sample_index);
    propagate_output_errors_back(solve_thread_index);
    calculate_weight_gradients(solve_thread_index, data_set.get_input_sample(sample_index));

    solver[solve_thread_index].reset();
  }
}

void Sparse_net_optimizer::calculate_output_errors(uint32 solve_thread_index, uint32 sample_index){
  uint32 neuron_index = net.neuron_array_size()-net.output_neuron_number();
  const uint32 neuron_number = 1 + static_cast<uint32>(net.neuron_array_size()/context.get_max_processing_threads());
  for(
    uint32 process_thread_index = 0;
    ( (process_thread_index < context.get_max_processing_threads())
      &&(static_cast<uint32>(net.neuron_array_size()) >= neuron_index) );
    ++process_thread_index
  ){ /* For every provided sample */
      process_threads[solve_thread_index].push_back(thread(
        &Sparse_net_optimizer::calculate_output_errors_thread, this, process_thread_index, sample_index,
        neuron_index, std::min(neuron_number, (net.neuron_array_size() - neuron_index))
      ));
      neuron_index += neuron_number;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::calculate_output_errors_thread(uint32 solve_thread_index, uint32 sample_index, uint32 neuron_index, uint32 neuron_number){
  sdouble32 buffer;

  if(neuron_index < (net.neuron_array_size() - net.output_neuron_number()))
    throw "Neuron index points to non-output Neuron in output layer error calculation!";

  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    /* Error =
     * (d cost over d feature) *
     * spike_function_derivative(neuron_memory_ilter) *
     * transfer_function_derivative(transfer_function_input)
     */
    buffer = cost_function->get_d_cost_over_d_feature(
      ((neuron_index + neuron_iterator) - (net.neuron_array_size() - net.output_neuron_number())),
      data_set.get_label_sample(sample_index), neuron_data[solve_thread_index]
    );
    buffer *= Spike_function::get_derivative(
      net.weight_table(net.neuron_array(neuron_index + neuron_iterator).memory_filter_idx()),
      transfer_function_input[solve_thread_index][neuron_index + neuron_iterator]
    );
    buffer *= transfer_function.get_derivative(
      net.neuron_array(neuron_index + neuron_iterator).transfer_function_idx(),
      transfer_function_output[solve_thread_index][neuron_index + neuron_iterator]
    );
    error_values[solve_thread_index][neuron_index + neuron_iterator]->store(buffer/static_cast<sdouble32>(data_set.get_number_of_samples()));
  }
}

void Sparse_net_optimizer::propagate_output_errors_back(uint32 solve_thread_index){
  uint32 synapses_iterator = 0;
  uint32 synapse_index_iterator = 0;
  uint32 process_thread_iterator;
  for(sint32 row_iterator = 0; row_iterator < gradient_step.cols_size(); ++row_iterator){
    process_thread_iterator = 0; /* Open up threads for the neurons in the same row */
    while(process_thread_iterator < gradient_step.cols(row_iterator)){
      synapses_iterator = 0;
      while(
        (context.get_max_processing_threads() > process_threads[solve_thread_index].size())
        &&(gradient_step.neuron_synapses_size() > static_cast<int>(synapses_iterator))
      ){
        if(
          (static_cast<uint32>(net.neuron_array_size()) > (gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator))
          &&(!Synapse_iterator::is_index_input(gradient_step.neuron_synapses(synapses_iterator).starts()))
        ){
          process_threads[solve_thread_index].push_back(thread(
            &Sparse_net_optimizer::backpropagation_thread, this, solve_thread_index,
            gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator
          ));
          ++process_thread_iterator;
        }
        ++synapse_index_iterator;
        if(synapse_index_iterator >= gradient_step.neuron_synapses(synapses_iterator).interval_size()){
          synapse_index_iterator = 0;
          ++synapses_iterator;
        }
      }/* while(context.get_max_processing_threads() > process_threads[solve_thread_index].size()) */
      wait_for_threads(process_threads[solve_thread_index]);
    }/* while(thread_iterator < gradient_step.cols(row_iterator)) */
  }
}

void Sparse_net_optimizer::calculate_weight_gradients(uint32 solve_thread_index, const vector<sdouble32>& input_sample){
  uint32 process_thread_iterator = 0;
  while(static_cast<int>(process_thread_iterator) < net.neuron_array_size()){
    while(
      (context.get_max_processing_threads() > process_threads[solve_thread_index].size())
      &&(net.neuron_array_size() > static_cast<int>(process_thread_iterator))
    ){
      process_threads[solve_thread_index].push_back(thread(
        &Sparse_net_optimizer::calculate_weight_gradients_thread, this, 
        solve_thread_index, ref(input_sample), process_thread_iterator
      ));

      ++process_thread_iterator;
    }/* while(context.get_max_processing_threads() > process_threads[solve_thread_index].size()) */

    wait_for_threads(process_threads[solve_thread_index]);
  } /* while(static_cast<int>(process_thread_iterator) < net.neuron_array_size()) */

}

void Sparse_net_optimizer::backpropagation_thread(uint32 solve_thread_index, uint32 neuron_index){
  sdouble32 buffer;
  sdouble32 addition;
  uint32 weight_index = 0;
  uint32 weight_synapse_index = 0;
  Synapse_iterator::iterate(net.neuron_array(neuron_index).input_indices(),[&](sint32 child_index){
    if(!Synapse_iterator::is_index_input(child_index)){
      buffer = *error_values[solve_thread_index][child_index];
      addition =
      *error_values[solve_thread_index][neuron_index]
      * net.weight_table(net.neuron_array(neuron_index).input_weights(weight_synapse_index).starts() + weight_index)
      * transfer_function.get_derivative(
        net.neuron_array(child_index).transfer_function_idx(),
        transfer_function_input[solve_thread_index][child_index]
      )
      * Spike_function::get_derivative(
        net.weight_table(net.neuron_array(child_index).memory_filter_idx()),
        transfer_function_output[solve_thread_index][child_index]
      );
      while(!error_values[solve_thread_index][child_index]->compare_exchange_weak(buffer, (buffer + addition)))
        buffer = *error_values[solve_thread_index][child_index];
    }
    ++weight_index;
    if(weight_index >= net.neuron_array(neuron_index).input_weights(weight_synapse_index).interval_size()){
      weight_index = 0;
      ++weight_synapse_index;
    }
  });
}

void Sparse_net_optimizer::calculate_weight_gradients_thread(uint32 solve_thread_index, const vector<sdouble32>& input_sample, uint32 neuron_index){
  sdouble32 buffer;
  sdouble32 addition;
  uint32 index;

  /* Calculate gradient for Bias (error * 1) */
  index = net.neuron_array(neuron_index).bias_idx();
  buffer = *weight_gradients[index];
  addition = (*error_values[solve_thread_index][neuron_index]);
  while(!weight_gradients[index]->compare_exchange_weak(buffer, buffer + addition))
    buffer = *weight_gradients[index];

  /* Calculate gradient for Memory filter (error * (1-memory_filter)) */
  /* maybe next time.. ? */

  /* Calculate gradient for each Weight (error * corresponding input) */
  const Neuron& neuron = net.neuron_array(neuron_index);
  uint32 weight_index = 0;
  uint32 weight_synapse_index = 0;
  sdouble32 neuron_input;
  Synapse_iterator::iterate(net.neuron_array(neuron_index).input_indices(),[&](sint32 child_index){
    if(Synapse_iterator::is_index_input(child_index))
      neuron_input = input_sample[Synapse_iterator::input_index_from_synapse_index(child_index)];
        else neuron_input = neuron_data[solve_thread_index][child_index];
    buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
    addition = neuron_input * *error_values[solve_thread_index][neuron_index];
    while(!weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index]->compare_exchange_weak(
      buffer, buffer + addition
    ))buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
    ++weight_index;
    if(weight_index >= neuron.input_weights(weight_synapse_index).interval_size()){
      weight_index = 0;
      ++weight_synapse_index;
    }
  });
}

} /* namespace sparse_net_library */
