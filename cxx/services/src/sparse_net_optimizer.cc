#include "services/sparse_net_optimizer.h"
#include "models/spike_function.h"
#include "services/neuron_router.h"

#include <atomic>
#include <thread>

namespace sparse_net_library{

void Sparse_net_optimizer::step(
  vector<vector<sdouble32>>& input_samples, 
  sdouble32 step_size_, uint32 sequence_size
){
  using std::atomic;
  using std::thread;

  if(0 != (input_samples.size() % sequence_size))
    throw "Number of samples are incompatible with Sequence size!";

  if(0 < step_size_)step_size = step_size;

  /* Calculate features for every input */
  vector<vector<sdouble32>> features = vector<vector<sdouble32>>(input_samples.size());
  uint32 sample_iterator = 0;
  vector<thread> calculate_threads;
  uint32 thread_iterator;
  
  for(unique_ptr<atomic<sdouble32>>& weight_gradient : weight_gradients) *weight_gradient = 0;

  for(vector<sdouble32> sample : input_samples){ /* For every provided sample */
    features[sample_iterator].reserve(input_samples[sample_iterator].size());
    features[sample_iterator] = solver.solve(input_samples[sample_iterator]);

    if(label_samples[sample_iterator].size() != features[sample_iterator].size())
      throw "Network output size doesn't match size of provided labels!";

    /* Calculate error value for each output Neurons */
    vector<sdouble32> transfer_function_input = solver.get_transfer_function_input();
    for(unique_ptr<atomic<sdouble32>>& error_value : error_values) *error_value = 0;
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
        sample_iterator, output_layer_iterator, features[sample_iterator]
      );
      buffer *= Spike_function::get_derivative(
        net.weight_table(net.neuron_array(neuron_iterator).memory_filter_idx())
      );
      buffer *= transfer_function.get_derivative(
        net.neuron_array(neuron_iterator).transfer_function_idx(),
        transfer_function_input[neuron_iterator]
      );
      *error_values[neuron_iterator] = buffer;
      ++output_layer_iterator;
    }

    /* Propagate error values back throughout the Neurons */
    uint32 synapses_iterator = 0;
    uint32 synapse_index_iterator = 0;
    for(sint32 row_iterator = 0; row_iterator < gradient_step.cols_size(); ++row_iterator){
      thread_iterator = 0; /* Open up threads for the neurons in the same row */
      while(thread_iterator < gradient_step.cols(row_iterator)){
        while(context.get_max_solve_threads() > calculate_threads.size()){
          calculate_threads.push_back(thread(
            &Sparse_net_optimizer::propagate_errors_back, this, 
            gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator
          ));

          ++thread_iterator;
          ++synapse_index_iterator;
          if(synapse_index_iterator >= gradient_step.neuron_synapses(synapses_iterator).interval_size()){
            synapse_index_iterator = 0; 
            ++synapses_iterator;
          }
        }/* while(context.get_max_solve_threads() > calculate_threads.size()) */

        while(0 < calculate_threads.size()){
          if(calculate_threads.back().joinable())
            calculate_threads.back().join();
        }
      }/* while(thread_iterator < gradient_step.cols(row_iterator)) */
    }

    /* Calculate gradient for each weight */  
    thread_iterator = 0;
    while(static_cast<int>(thread_iterator) < net.neuron_array_size()){
      while(context.get_max_solve_threads() > calculate_threads.size()){
        calculate_threads.push_back(thread(
          &Sparse_net_optimizer::calculate_weight_gradients, this, 
          thread_iterator
        ));

        ++thread_iterator;
      }/* while(context.get_max_solve_threads() > calculate_threads.size()) */

      while(0 < calculate_threads.size()){
        if(calculate_threads.back().joinable())
          calculate_threads.back().join();
      }
    }

    ++sample_iterator;
    if(0 == (sample_iterator % sequence_size)) /* In case its the end of one sample */
      solver.reset();
  }

  /* Update the weights with the gradients */
  thread_iterator = 0;
  while(static_cast<int>(thread_iterator) < net.weight_table_size()){
    while(context.get_max_solve_threads() > calculate_threads.size()){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::update_weights_with_gradients, this, 
        thread_iterator
      ));

      ++thread_iterator;
    }/* while(context.get_max_solve_threads() > calculate_threads.size()) */

    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable())
        calculate_threads.back().join();
    }
  }

}

void Sparse_net_optimizer::calculate_weight_gradients(uint32 neuron_index){
  sdouble32 buffer;
  sdouble32 new_value;
  uint32 index;

  /* Calculate gradient for Bias (error * 1) */
  index = net.neuron_array(neuron_index).bias_idx();
  buffer = *weight_gradients[index];
  new_value = (buffer + (*error_values[neuron_index] / static_cast<sdouble32>(label_samples.size())));
  while(!weight_gradients[index]->compare_exchange_weak(buffer, new_value)){
    buffer = *weight_gradients[index];
    new_value = (buffer + (*error_values[neuron_index] / static_cast<sdouble32>(label_samples.size())));
  }

  /* Calculate gradient for Memory filter (error * (1-memory_filter)) */
  /* maybe next time.. ? */

  /* Calculate gradient for each Weight (error * corresponding input) */
  const Neuron& neuron = net.neuron_array(neuron_index);
  uint32 weight_index = 0;
  uint32 weight_synapse_index = 0;
  neuron_router.run_for_neuron_inputs(neuron_index,[&](sint32 child_index){
    buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
    new_value = buffer 
      + (solver.get_neuron_data(child_index) * *error_values[neuron_index]  / static_cast<sdouble32>(label_samples.size()));
    while(!weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index]->compare_exchange_weak(
      buffer, new_value
    )){
      buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
      new_value = buffer 
        + (solver.get_neuron_data(child_index) * *error_values[neuron_index]  / static_cast<sdouble32>(label_samples.size()));
    }
    ++weight_index; 
    if(weight_index >= neuron.input_weights(weight_synapse_index).interval_size()){
      weight_index = 0; 
      ++weight_synapse_index;
    }
  });
}

} /* namespace sparse_net_library */