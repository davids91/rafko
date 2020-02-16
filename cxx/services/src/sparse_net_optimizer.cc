#include "services/sparse_net_optimizer.h"
#include "models/spike_function.h"
#include "services/neuron_router.h"

#include <atomic>
#include <thread>

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;

void Sparse_net_optimizer::step(
  vector<vector<sdouble32>>& input_samples,
  sdouble32 step_size_, uint32 sequence_size
){
  if(0 != (input_samples.size() % sequence_size))
    throw "Number of samples are incompatible with Sequence size!";

  if(0 < step_size_)step_size = step_size;
  last_error.store(0);

  /* Calculate features for every input */
  vector<thread> calculate_threads;
  uint32 process_thread_iterator;

  for(unique_ptr<atomic<sdouble32>>& weight_gradient : weight_gradients[0]) *weight_gradient = 0;
  for(uint32 sample_iterator = 0; sample_iterator < input_samples.size();sample_iterator += context.get_max_solve_threads()){ /* For every provided sample */
    process_thread_iterator = 0;
    while(
      (context.get_max_solve_threads() > calculate_threads.size())
      &&(error_values.size() > process_thread_iterator)
      &&((sample_iterator + process_thread_iterator) < input_samples.size())
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::calculate_gradient, this,
        ref(input_samples[sample_iterator]), (sample_iterator + process_thread_iterator), process_thread_iterator
      ));
      ++process_thread_iterator;
    }/* while(context.get_max_processing_threads() > calculate_threads.size()) */

    process_thread_iterator = 0;
    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  }

  /* Update the weights with the gradients */
  process_thread_iterator = 0;
  while(static_cast<int>(process_thread_iterator) < net.weight_table_size()){
    while(
      (context.get_max_processing_threads() > calculate_threads.size())
      &&(net.weight_table_size() > static_cast<int>(process_thread_iterator))
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::update_weights_with_gradients, this,
        process_thread_iterator, 0
      ));
      ++process_thread_iterator;
    }/* while(context.get_max_processing_threads() > calculate_threads.size()) */

    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  }

  /* Update the weights in the Solution as well */
  process_thread_iterator = 0;
  uint32 neuron_weight_synapse_starts = 0;
  uint32 inner_neuron_weight_index_starts = 0;
  for(sint32 partial_index = 0; partial_index < net_solution.partial_solutions_size(); ++partial_index){
    Partial_solution& partial = *net_solution.mutable_partial_solutions(partial_index);
    process_thread_iterator = 0;
    neuron_weight_synapse_starts = 0;
    inner_neuron_weight_index_starts = 0;
    while(
      (context.get_max_processing_threads() > calculate_threads.size())
      &&(process_thread_iterator < partial.internal_neuron_number())
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::copy_weight_to_solution, this, process_thread_iterator,
        ref(partial), neuron_weight_synapse_starts, inner_neuron_weight_index_starts
      ));
      inner_neuron_weight_index_starts += 2; /* bias and memory filter */
      for(uint32 i = 0; i < partial.weight_synapse_number(process_thread_iterator); ++i){
        inner_neuron_weight_index_starts +=
          partial.weight_indices(neuron_weight_synapse_starts + i).interval_size();
      }
      neuron_weight_synapse_starts += partial.weight_synapse_number(process_thread_iterator);
      ++process_thread_iterator;
    }
    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  } /* for(uint32 partial_index = 0;partial_index < net_solution.partial_solutions_size(); ++partial_index) */
}

void Sparse_net_optimizer::calculate_gradient(vector<sdouble32>& input_sample, uint32 sample_iterator, uint32 solve_thread_index){

  /* Solve the network for the given input */
  feature_buffers[solve_thread_index].reserve(label_samples[sample_iterator].size());
  feature_buffers[solve_thread_index].clear();
  feature_buffers[solve_thread_index] = solver[solve_thread_index].solve(input_sample);
  if(label_samples[sample_iterator].size() != feature_buffers[solve_thread_index].size())
    throw "Network output size doesn't match size of provided labels!";

  vector<thread> calculate_threads;
  vector<sdouble32> transfer_function_input = solver[solve_thread_index].get_transfer_function_input();

  /* Calculate error value for each output Neurons */
  for(unique_ptr<atomic<sdouble32>>& error_value : error_values[solve_thread_index]) *error_value = 0;
  uint32 output_layer_iterator = 0;
  sdouble32 buffer;
  for( /* For every ouput layer Neuron */
    sint32 neuron_iterator = net.neuron_array_size()-net.output_neuron_number();
    neuron_iterator < net.neuron_array_size();
    ++neuron_iterator
  ){ /* Set its error value */
    /* Error =
     * (d cost over d feature) *
     * spike_function_derivative(neuron_memory_ilter) *
     * transfer_function_derivative(transfer_function_input)
     */
    buffer = cost_function->get_d_cost_over_d_feature(
      sample_iterator, output_layer_iterator, feature_buffers[solve_thread_index]
    );
    buffer *= Spike_function::get_derivative(
      net.weight_table(net.neuron_array(neuron_iterator).memory_filter_idx()),
      transfer_function.get_derivative(
        net.neuron_array(neuron_iterator).transfer_function_idx(),
        transfer_function_input[neuron_iterator+net.output_neuron_number()-net.neuron_array_size()]
      )
    );
    buffer *= transfer_function.get_derivative(
      net.neuron_array(neuron_iterator).transfer_function_idx(),
      transfer_function_input[neuron_iterator+net.output_neuron_number()-net.neuron_array_size()]
    );
    *error_values[solve_thread_index][neuron_iterator] = buffer;
    buffer = last_error; /* Summarize the currently calculated error value */
    while(!last_error.compare_exchange_weak(
      buffer,(buffer + *error_values[solve_thread_index][neuron_iterator]/static_cast<sdouble32>(label_samples.size()))
    ))buffer = last_error;
    ++output_layer_iterator;
  }

  /* Propagate error values back throughout the Neurons */
  uint32 synapses_iterator = 0;
  uint32 synapse_index_iterator = 0;
  uint32 process_thread_iterator;
  for(sint32 row_iterator = 0; row_iterator < gradient_step.cols_size(); ++row_iterator){
    process_thread_iterator = 0; /* Open up threads for the neurons in the same row */
    while(process_thread_iterator < gradient_step.cols(row_iterator)){
      synapses_iterator = 0;
      while(
        (context.get_max_processing_threads() > calculate_threads.size())
        &&(gradient_step.neuron_synapses_size() > static_cast<int>(synapses_iterator))
      ){
        calculate_threads.push_back(thread(
          &Sparse_net_optimizer::propagate_errors_back, this,
          gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator,
          solve_thread_index
        ));
        ++process_thread_iterator;
        ++synapse_index_iterator;
        if(synapse_index_iterator >= gradient_step.neuron_synapses(synapses_iterator).interval_size()){
          synapse_index_iterator = 0;
          ++synapses_iterator;
        }
      }/* while(context.get_max_processing_threads() > calculate_threads.size()) */
      while(0 < calculate_threads.size()){
        if(calculate_threads.back().joinable()){
          calculate_threads.back().join();
          calculate_threads.pop_back();
        }
      }
    }/* while(thread_iterator < gradient_step.cols(row_iterator)) */
  }

  /* Calculate gradient for each weight */
  process_thread_iterator = 0;
  while(static_cast<int>(process_thread_iterator) < net.neuron_array_size()){
    while(
      (context.get_max_processing_threads() > calculate_threads.size())
      &&(net.neuron_array_size() > static_cast<int>(process_thread_iterator))
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::calculate_weight_gradients, this,
        ref(input_sample), process_thread_iterator, solve_thread_index
      ));

      ++process_thread_iterator;
    }/* while(context.get_max_processing_threads() > calculate_threads.size()) */

    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  } /* while(static_cast<int>(process_thread_iterator) < net.neuron_array_size()) */

  solver[solve_thread_index].reset();
}

void Sparse_net_optimizer::copy_weight_to_solution(
  uint32 inner_neuron_index, /* In the solution, the weights are copied in without optimization */
  Partial_solution& partial,
  uint32 neuron_weight_synapse_starts,
  uint32 inner_neuron_weight_index_starts /* In the solution, the weights are copied in without optimization */
){ /*!Note: After shared weight optimization, this part is to be re-worked */
    uint32 weights_copied = 0;
    uint32 weight_interval_index = 0;
    uint32 weight_synapse_index = 0;
    Neuron& neuron = *net.mutable_neuron_array(partial.actual_index(inner_neuron_index));
    partial.set_weight_table(partial.bias_index(inner_neuron_index),net.weight_table(neuron.bias_idx()));
    partial.set_weight_table(partial.memory_filter_index(inner_neuron_index),net.weight_table(neuron.memory_filter_idx()));
    weights_copied += 2;
    neuron_router.run_for_neuron_inputs(partial.actual_index(inner_neuron_index),[&](sint32 child_index){
      partial.set_weight_table(
        (inner_neuron_weight_index_starts + weights_copied),
        net.weight_table(neuron.input_weights(weight_synapse_index).starts() + weight_interval_index)
      );
      ++weights_copied;
      ++weight_interval_index;
      if(weight_interval_index >= neuron.input_weights(weight_synapse_index).interval_size()){
        weight_interval_index = 0;
        ++weight_synapse_index;
      }
    });
  }

void Sparse_net_optimizer::calculate_weight_gradients(vector<sdouble32>& input_sample, uint32 neuron_index, uint32 solve_thread_index){
  sdouble32 buffer;
  sdouble32 addition;
  uint32 index;

  /* Calculate gradient for Bias (error * 1) */
  index = net.neuron_array(neuron_index).bias_idx();
  buffer = *weight_gradients[solve_thread_index][index];
  addition = (*error_values[solve_thread_index][neuron_index]) * step_size;
  while(!weight_gradients[solve_thread_index][index]->compare_exchange_weak(buffer, buffer + addition))
    buffer = *weight_gradients[solve_thread_index][index];

  /* Calculate gradient for Memory filter (error * (1-memory_filter)) */
  /* maybe next time.. ? */

  /* Calculate gradient for each Weight (error * corresponding input) */
  const Neuron& neuron = net.neuron_array(neuron_index);
  uint32 weight_index = 0;
  uint32 weight_synapse_index = 0;
  sdouble32 neuron_input;
  neuron_router.run_for_neuron_inputs(neuron_index,[&](sint32 child_index){
    if(Synapse_iterator::is_index_input(child_index))
      neuron_input = input_sample[Synapse_iterator::input_index_from_synapse_index(child_index)];
      else neuron_input = solver[solve_thread_index].get_neuron_data(child_index);
    buffer = *weight_gradients[solve_thread_index][neuron.input_weights(weight_synapse_index).starts() + weight_index];
    addition = neuron_input * *error_values[solve_thread_index][neuron_index] * step_size;
    while(!weight_gradients[solve_thread_index][neuron.input_weights(weight_synapse_index).starts() + weight_index]->compare_exchange_weak(
      buffer, buffer + addition
    ))buffer = *weight_gradients[solve_thread_index][neuron.input_weights(weight_synapse_index).starts() + weight_index];
    ++weight_index;
    if(weight_index >= neuron.input_weights(weight_synapse_index).interval_size()){
      weight_index = 0;
      ++weight_synapse_index;
    }
  });
}

} /* namespace sparse_net_library */
