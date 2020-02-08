#include "services/sparse_net_optimizer.h"
#include "models/spike_function.h"
#include "services/neuron_router.h"

#include <atomic>
#include <thread>

namespace sparse_net_library{

sdouble32 Sparse_net_optimizer::last_error(){
  sdouble32 score = 0;
  for( /* For every ouput layer Neuron */
    sint32 neuron_iterator = net.neuron_array_size()-net.output_neuron_number(); 
    neuron_iterator < net.neuron_array_size();
    ++neuron_iterator
  )score += *error_values[neuron_iterator];
  return score;
}

void Sparse_net_optimizer::step(
  vector<vector<sdouble32>>& input_samples, 
  sdouble32 step_size_, uint32 sequence_size
){
  using std::atomic;
  using std::thread;
  using std::ref;

  if(0 != (input_samples.size() % sequence_size))
    throw "Number of samples are incompatible with Sequence size!";

  if(0 < step_size_)step_size = step_size;

  /* Calculate features for every input */
  vector<vector<sdouble32>> features = vector<vector<sdouble32>>(input_samples.size());
  uint32 sample_iterator = 0;
  vector<thread> calculate_threads;
  uint32 thread_iterator;
  
  for(unique_ptr<atomic<sdouble32>>& weight_gradient : weight_gradients) *weight_gradient = 0;
  for(vector<sdouble32>& input_sample : input_samples){ /* For every provided sample */
    features[sample_iterator].reserve(input_samples[sample_iterator].size());
    features[sample_iterator] = solver.solve(input_samples[sample_iterator]);
    if(label_samples[sample_iterator].size() != features[sample_iterator].size()){
      throw "Network output size doesn't match size of provided labels!";
    }

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
        net.weight_table(net.neuron_array(neuron_iterator).memory_filter_idx()),
        transfer_function.get_derivative(
          net.neuron_array(neuron_iterator).transfer_function_idx(),
          transfer_function_input[neuron_iterator]
        )
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
        synapses_iterator = 0;
        while(
          (context.get_max_solve_threads() > calculate_threads.size())
          &&(gradient_step.neuron_synapses_size() > static_cast<int>(synapses_iterator))
        ){
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
          if(calculate_threads.back().joinable()){
            calculate_threads.back().join();
            calculate_threads.pop_back();
          }
        }
      }/* while(thread_iterator < gradient_step.cols(row_iterator)) */
    }
    /* Calculate gradient for each weight */  
    thread_iterator = 0;
    while(static_cast<int>(thread_iterator) < net.neuron_array_size()){
      while(
        (context.get_max_solve_threads() > calculate_threads.size())
        &&(net.neuron_array_size() > static_cast<int>(thread_iterator))
      ){
        calculate_threads.push_back(thread(
          &Sparse_net_optimizer::calculate_weight_gradients, this, 
          thread_iterator, std::ref(input_sample)
        ));

        ++thread_iterator;
      }/* while(context.get_max_solve_threads() > calculate_threads.size()) */

      while(0 < calculate_threads.size()){
        if(calculate_threads.back().joinable()){
          calculate_threads.back().join();
          calculate_threads.pop_back();
        }
      }
    }
    ++sample_iterator;
    if(0 == (sample_iterator % sequence_size)) /* In case its the end of one sample */
      solver.reset();
  }
  #if 0
  std::cout << "Error["<< *error_values[0] 
  <<"]";//<<"] ==>("
  <<"Gradient["<< *weight_gradients[0]
  <<"]; Gradient["<< *weight_gradients[2]
  <<"]; Gradient["<< *weight_gradients[3] <<"]"
  ")                                            ";
  std::cout << std::endl;
  //std::cin.get();
  #endif
  /* Update the weights with the gradients */
  thread_iterator = 0;
  while(static_cast<int>(thread_iterator) < net.weight_table_size()){
    while(
      (context.get_max_solve_threads() > calculate_threads.size())
      &&(net.weight_table_size() > static_cast<int>(thread_iterator))
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::update_weights_with_gradients, this, 
        thread_iterator
      ));
      ++thread_iterator;
    }/* while(context.get_max_solve_threads() > calculate_threads.size()) */

    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  }

  /* Update the weights in the Solution as well */
  thread_iterator = 0; /* Open up threads for the neurons */
  uint32 neuron_weight_synapse_starts = 0;
  uint32 inner_neuron_weight_index_starts = 0;
  for(sint32 partial_index = 0; partial_index < net_solution.partial_solutions_size(); ++partial_index){
    Partial_solution& partial = *net_solution.mutable_partial_solutions(partial_index);
    thread_iterator = 0;
    neuron_weight_synapse_starts = 0;
    inner_neuron_weight_index_starts = 0;
    while(
      (context.get_max_processing_threads() > calculate_threads.size())
      &&(thread_iterator < partial.internal_neuron_number())
    ){
      calculate_threads.push_back(thread(
        &Sparse_net_optimizer::copy_weight_to_solution, this, thread_iterator,
        ref(partial), neuron_weight_synapse_starts, inner_neuron_weight_index_starts
      ));
      inner_neuron_weight_index_starts += 2; /* bias and memory filter */
      for(uint32 i = 0; i < partial.weight_synapse_number(thread_iterator); ++i){
        inner_neuron_weight_index_starts += 
          partial.weight_indices(neuron_weight_synapse_starts + i).interval_size();
      }
      neuron_weight_synapse_starts += partial.weight_synapse_number(thread_iterator);
      ++thread_iterator;
    }
    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  } /* for(uint32 partial_index = 0;partial_index < net_solution.partial_solutions_size(); ++partial_index) */
  #if 0
  std::cout << "\nWeights:" << std::endl;
  for(sint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
    std::cout << "["<< net.weight_table(weight_index) << "]";
  }
  std::cout << std::endl;
  #endif
  #if 0
  std::cout << "\t\t";
  for(sint32 weight_index = 0; weight_index < net_solution.partial_solutions(0).weight_table_size(); ++weight_index){
    std::cout << "["<< net_solution.partial_solutions(0).weight_table(weight_index) << "]                             ";
  }
  //std::cout << std::endl;
  #endif
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


void Sparse_net_optimizer::calculate_weight_gradients(uint32 neuron_index, vector<sdouble32>& input_sample){
  sdouble32 buffer;
  sdouble32 new_value;
  uint32 index;

  /* Calculate gradient for Bias (error * 1) */
  index = net.neuron_array(neuron_index).bias_idx();
  buffer = *weight_gradients[index];
  new_value = (buffer + (*error_values[neuron_index])) * step_size;
  while(!weight_gradients[index]->compare_exchange_weak(buffer, new_value)){
    buffer = *weight_gradients[index];
    new_value = (buffer + (*error_values[neuron_index])) * step_size;
  }

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
      else neuron_input = solver.get_neuron_data(child_index);
    buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
    new_value = buffer 
      + (neuron_input * *error_values[neuron_index]) * step_size;
    while(!weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index]->compare_exchange_weak(
      buffer, new_value
    )){
      buffer = *weight_gradients[neuron.input_weights(weight_synapse_index).starts() + weight_index];
      new_value = buffer 
        + (neuron_input * *error_values[neuron_index]) * step_size;
    }
    ++weight_index; 
    if(weight_index >= neuron.input_weights(weight_synapse_index).interval_size()){
      weight_index = 0; 
      ++weight_synapse_index;
    }
  });
}

} /* namespace sparse_net_library */