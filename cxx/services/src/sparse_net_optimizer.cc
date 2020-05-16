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

#include <atomic>
#include <thread>
#include <cmath>
#include <stdexcept>

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;
using std::min;
using std::max;

void Sparse_net_optimizer::step(void){
  for(unique_ptr<atomic<sdouble32>>& weight_gradient : get_weight_gradient())
    weight_gradient->store(0);
  weight_updater->start();
  while(!weight_updater->is_finished()){
    for(
      uint32 thread_index = 0;
      thread_index < min(context.get_minibatch_size(),static_cast<uint32>(context.get_max_solve_threads()));
      ++thread_index
    )solve_threads.push_back(thread(
      &Sparse_net_optimizer::step_thread, this, thread_index,
      max(1u,(context.get_minibatch_size()/context.get_max_solve_threads()))
    ));
    wait_for_threads(solve_threads);
    normalize_weight_gradients();
    weight_updater->iterate(get_weight_gradient(), *net_solution); /* Update the weights of the SparseNet and the solution */
  } /* while(!weight_updater->finished()) */
  ++loops_unchecked;

  if(loops_unchecked > std::min(double_literal(50.0),(test_set.get_error()/context.get_step_size()))){
    uint32 sample_start_index = 0;
    const uint32 samples_to_evaluate = 1 + static_cast<uint32>(train_set.get_number_of_sequences()/context.get_max_solve_threads());
    for(
      uint32 thread_index = 0;
      thread_index < min(test_set.get_number_of_sequences(),static_cast<uint32>(context.get_max_solve_threads()));
      ++thread_index
    ){
      solve_threads.push_back(thread(
        &Sparse_net_optimizer::evaluate_thread, this, thread_index, sample_start_index,
        min(samples_to_evaluate,(train_set.get_number_of_sequences() - sample_start_index))
      ));
      sample_start_index += samples_to_evaluate;
    }
    wait_for_threads(solve_threads);
    loops_unchecked = 0;
  }
}

void Sparse_net_optimizer::evaluate_thread(uint32 solve_thread_index, uint32 sample_start, uint32 samples_to_evaluate){
  if(test_set.get_feature_size() != solvers[solve_thread_index]->get_output_size())
    throw std::runtime_error("Network output size doesn't match size of provided testing labels!");
  for(uint32 sample_iterator = 0; sample_iterator < samples_to_evaluate; ++sample_iterator){
    for(uint32 sequence_iterator = 0; sequence_iterator < train_set.get_sequence_size(); ++sequence_iterator){
      solvers[solve_thread_index]->solve(test_set.get_input_sample(sample_start + sample_iterator + sequence_iterator)); /* Solve the network for the sampled labels input */
      test_set.set_feature_for_label(
        (sample_start + sample_iterator + sequence_iterator), solvers[solve_thread_index]->get_neuron_memory().get_const_element(0)
      ); /* Re-calculate error for the training set */
    }
    solvers[solve_thread_index]->reset();
  }
}

void Sparse_net_optimizer::step_thread(uint32 solve_thread_index, uint32 samples_to_evaluate){
  uint32 sample_index;

  if(train_set.get_feature_size() != solvers[solve_thread_index]->get_output_size())
    throw std::runtime_error("Network output size doesn't match size of provided training labels!");

  for(uint32 sample = 0; sample < samples_to_evaluate; ++sample){
    sample_index = (rand()%(train_set.get_number_of_sequences())) * train_set.get_sequence_size();

    /* Evaluate the current sequence step by step */
    solvers[solve_thread_index]->reset();
    for(uint32 sequence_iterator = 0; sequence_iterator < train_set.get_sequence_size(); ++sequence_iterator){
      neuron_data_sequences[solve_thread_index].step();
      solvers[solve_thread_index]->solve(train_set.get_input_sample(sample_index)); /* Solve the network for the sampled labels input */
      transfer_function_input[solve_thread_index][sequence_iterator] = solvers[solve_thread_index]->get_transfer_function_input();
      neuron_data_sequences[solve_thread_index].copy_latest(solvers[solve_thread_index]->get_neuron_memory());
      train_set.set_feature_for_label(sample_index, neuron_data_sequences[solve_thread_index].get_const_element(0)); /* Re-calculate error for the training set */
      /* Only calculate the derivatives for the first un-truncated sequences */
      if(sequence_iterator < sequence_truncation){ /* Since the network will be the same, the derivatives can be re-used for the later sequences */
        for(unique_ptr<atomic<sdouble32>>& derivative_value : weight_derivatives[solve_thread_index][sequence_iterator]) *derivative_value = 0;
        calculate_derivatives(solve_thread_index, sequence_iterator, sample_index);
      }
      ++sample_index;
    }

    /* Calculate the gradients from the current sequence */
    for(sint32 sequence_iterator = train_set.get_sequence_size()-1; sequence_iterator >= 0 ; --sequence_iterator){
      --sample_index;

      for(unique_ptr<atomic<sdouble32>>& error_value : error_values[solve_thread_index])
        *error_value = 0;

      calculate_output_errors(solve_thread_index, sequence_iterator, sample_index);
      propagate_output_errors_back(solve_thread_index, sequence_iterator);
      accumulate_weight_gradients(solve_thread_index, sequence_iterator, sample_index);
    }
    solvers[solve_thread_index]->reset();
    neuron_data_sequences[solve_thread_index].reset();
  }
}

void Sparse_net_optimizer::calculate_derivatives(uint32 solve_thread_index, uint32 sequence_index, uint32 sample_index){
  uint32 neuron_index = net.neuron_array_size()-net.output_neuron_number(); /* Start from the output layer */
  const uint32 neuron_number = 1 + (net.output_neuron_number()/static_cast<uint32>(context.get_max_processing_threads()));
  for( /* As long as there are free thread-slots or remaining neurons to be processed.. */
    uint32 process_thread_index = 0;
    process_thread_index < min(static_cast<uint32>(context.get_max_processing_threads()),static_cast<uint32>(net.output_neuron_number()));
    ++process_thread_index
  ){ /* Iterate through the neurons in the network */
    process_threads[solve_thread_index].push_back(thread(
      &Sparse_net_optimizer::calculate_derivatives_thread, this, solve_thread_index, sample_index, sequence_index,
      neuron_index, min(neuron_number, (net.output_neuron_number() - neuron_index))
    ));
    neuron_index += neuron_number;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::calculate_derivatives_thread(uint32 solve_thread_index, uint32 sample_index, uint32 sequence_index, uint32 neuron_index, uint32 neuron_number){
  sdouble32 buffer;
  sdouble32 addition;
  uint32 input_index_offset = 0;
  uint32 input_synapse_index = 0;

  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    input_index_offset = 0;
    input_synapse_index = 0;
    Synapse_iterator<>::iterate(net.neuron_array(neuron_iterator).input_weights(),[&](
      Index_synapse_interval weight_synapse, sint32 weight_index
    ){
      if(static_cast<sint32>(input_synapse_index) < net.neuron_array(neuron_iterator).input_indices_size()){
        if( /* in case this input is from the past */
          (0 < net.neuron_array(neuron_iterator).input_indices(input_synapse_index).reach_past_loops())
          &&(net.neuron_array(neuron_iterator).input_indices(input_synapse_index).reach_past_loops() <= sequence_index)
        ){ /* but that is included in the current sequence */
          if((net.neuron_array(neuron_iterator).input_indices(input_synapse_index).starts() + input_index_offset) == neuron_iterator){ /* The past input is from itself.. */
            Synapse_iterator<>::iterate(net.neuron_array(neuron_iterator).input_weights(),[&](
              Index_synapse_interval weight_synapse, sint32 child_weight_index
            ){ /* go through all the weights of the Neuron */
              if(child_weight_index != static_cast<sint32>(neuron_iterator)){ /* don't update own gradient */
                /* And for every weight add the corresponding gradient in the previous sequence into the current one */
                addition = get_derivative_for(
                  solve_thread_index, sequence_index, child_weight_index,
                  net.neuron_array(neuron_iterator).input_indices(input_synapse_index)
                ) * net.weight_table(weight_index);
                while(!weight_derivatives[solve_thread_index][sequence_index][child_weight_index]
                  ->compare_exchange_weak(buffer, (buffer + addition))
                )buffer = *weight_derivatives[solve_thread_index][sequence_index][child_weight_index];
              }
            });
          }else throw std::runtime_error("Optimizer doesn't support input types from the past of other Neurons!");
        } /* Neuron input is from the past! */

        if(Synapse_iterator<>::is_index_input(net.neuron_array(neuron_iterator).input_indices(input_synapse_index).starts())){ /* Neuron input from a sample! */
          addition = train_set.get_input_sample(sample_index)[Synapse_iterator<>::input_index_from_synapse_index(
            net.neuron_array(neuron_iterator).input_indices(input_synapse_index).starts() - input_index_offset
          )];
        }else{ /* Neuron input from another internal Neuron input! */
          addition = neuron_data_sequences[solve_thread_index].get_const_element(
            net.neuron_array(neuron_iterator).input_indices(input_synapse_index).reach_past_loops()
          )[net.neuron_array(neuron_iterator).input_indices(input_synapse_index).starts() + input_index_offset];
        }
        ++input_index_offset;
        if(net.neuron_array(neuron_iterator).input_indices(input_synapse_index).interval_size() <= input_index_offset){
          input_index_offset = 0;
          ++input_synapse_index;
        }
      }else addition = 1; /* Bias! */
      while(
        !weight_derivatives[solve_thread_index][sequence_index][weight_index]->compare_exchange_weak(buffer, (buffer + addition))
      )buffer = *weight_derivatives[solve_thread_index][sequence_index][weight_index];
    });
  }
}

void Sparse_net_optimizer::calculate_output_errors(uint32 solve_thread_index, uint32 sequence_index, uint32 sample_index){
  uint32 neuron_index = net.neuron_array_size()-net.output_neuron_number(); /* Start from the output layer */
  const uint32 neuron_number = 1 + (net.output_neuron_number()/static_cast<uint32>(context.get_max_processing_threads()));
  for( /* As long as there are free thread-slots or remaining neurons to be processed.. */
    uint32 process_thread_index = 0;
    process_thread_index < min(static_cast<uint32>(context.get_max_processing_threads()),static_cast<uint32>(net.output_neuron_number()));
    ++process_thread_index
  ){ /* Iterate through the neurons in the network */
    process_threads[solve_thread_index].push_back(thread(
      &Sparse_net_optimizer::calculate_output_errors_thread, this, solve_thread_index, sequence_index, sample_index,
      neuron_index, min(neuron_number, (net.output_neuron_number() - neuron_index))
    ));
    neuron_index += neuron_number;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::calculate_output_errors_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 sample_index, uint32 neuron_index, uint32 neuron_number){
  sdouble32 buffer;
  sdouble32 addition;
  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    addition = cost_function->get_d_cost_over_d_feature(
      ((neuron_index + neuron_iterator) - (net.neuron_array_size() - net.output_neuron_number())),
      train_set.get_label_sample(sample_index),
      neuron_data_sequences[solve_thread_index].get_const_element(sequence_index,Input_synapse_interval())
    ) * transfer_function.get_derivative(
      net.neuron_array(neuron_index + neuron_iterator).transfer_function_idx(),
      transfer_function_input[solve_thread_index][sequence_index][neuron_index + neuron_iterator]
    );

    while(!error_values[solve_thread_index][neuron_index + neuron_iterator]->compare_exchange_weak(buffer, (buffer + addition)))
     buffer = *error_values[solve_thread_index][neuron_index + neuron_iterator];
  }
}

void Sparse_net_optimizer::propagate_output_errors_back(uint32 solve_thread_index, uint32 sequence_index){
  uint32 synapses_iterator = 0;
  uint32 synapse_index_iterator = 0;
  uint32 process_thread_iterator;
  /*!Note: Neurons in the same row can be done in paralell, as they are no interlapping dependencies inbetween them */
  for(sint32 row_iterator = 0; row_iterator < gradient_step.cols_size(); ++row_iterator){
    process_thread_iterator = 0; /* Open up threads for the neurons in the same row */
    while(process_thread_iterator < gradient_step.cols(row_iterator)){
      synapses_iterator = 0;
      while( /* Until there are available threads to open and remaining neurons in the current row */
        (context.get_max_processing_threads() > process_threads[solve_thread_index].size())
        &&(gradient_step.neuron_synapses_size() > static_cast<int>(synapses_iterator))
      ){
        if( /* In case the current synapse index points to a Neuron in the network */
          (static_cast<uint32>(net.neuron_array_size()) > (gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator))
          &&(!Synapse_iterator<>::is_index_input(gradient_step.neuron_synapses(synapses_iterator).starts()))
        ){ /* And the current synapse index is not pointing to an input */
          process_threads[solve_thread_index].push_back(thread(
            &Sparse_net_optimizer::backpropagation_thread, this, solve_thread_index, sequence_index,
            gradient_step.neuron_synapses(synapses_iterator).starts() + synapse_index_iterator
          ));
          ++process_thread_iterator;
        }
        ++synapse_index_iterator;
        if(synapse_index_iterator >= gradient_step.neuron_synapses(synapses_iterator).interval_size()){
          synapse_index_iterator = 0;
          ++synapses_iterator;
        }
      }/* while((context.get_max_processing_threads() > process_threads[solve_thread_index].size())&&...) */
      wait_for_threads(process_threads[solve_thread_index]);
    }/* while(thread_iterator < gradient_step.cols(row_iterator)) */
  }
}

void Sparse_net_optimizer::backpropagation_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 neuron_index){
  sdouble32 buffer;
  sdouble32 addition;
  uint32 weight_index = 0;
  uint32 weight_synapse_index = 0;
  Synapse_iterator<Input_synapse_interval>::iterate(net.neuron_array(neuron_index).input_indices(),[&](
    Input_synapse_interval input_synapse, sint32 child_index
  ){
    if(
      (!Synapse_iterator<>::is_index_input(child_index))
      &&(input_synapse.reach_past_loops() <= sequence_index)
    ){ /* Calculate the value to add to the child's error, then try to add to it */
      buffer = *error_values[solve_thread_index][child_index];
      addition = *error_values[solve_thread_index][neuron_index]
        * net.weight_table(net.neuron_array(neuron_index).input_weights(weight_synapse_index).starts() + weight_index)
        * transfer_function.get_derivative(
          net.neuron_array(child_index).transfer_function_idx(),
          transfer_function_input[solve_thread_index][sequence_index - input_synapse.reach_past_loops()][child_index]
        );
      while(!error_values[solve_thread_index][child_index]
        ->compare_exchange_weak(buffer, (buffer + addition))
      )buffer = *error_values[solve_thread_index][child_index];
    }
    ++weight_index;
    if(weight_index >= net.neuron_array(neuron_index).input_weights(weight_synapse_index).interval_size()){
      weight_index = 0;
      ++weight_synapse_index;
    }
  });
}

void Sparse_net_optimizer::accumulate_weight_gradients(uint32 solve_thread_index, uint32 sequence_index, uint32 sample_index){
  uint32 neuron_iterator = 0;
  while(static_cast<sint32>(neuron_iterator) < net.neuron_array_size()){
    while( /* As long as there are remaining threads to open */
      (context.get_max_processing_threads() > process_threads[solve_thread_index].size())
      &&(net.neuron_array_size() > static_cast<int>(neuron_iterator))
    ){ /* And the thread would process an existing Neuron */
      process_threads[solve_thread_index].push_back(thread(
        &Sparse_net_optimizer::accumulate_weight_gradients_thread, this, 
        solve_thread_index, sequence_index, sample_index, neuron_iterator
      ));
      ++neuron_iterator;
    }/* while((context.get_max_processing_threads() > process_threads[solve_thread_index].size()))&&... */
    wait_for_threads(process_threads[solve_thread_index]);
  } /* while(static_cast<int>(neuron_iterator) < net.neuron_array_size()) */
}

void Sparse_net_optimizer::accumulate_weight_gradients_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 sample_index, uint32 neuron_index){
  sdouble32 buffer;
  sdouble32 addition;

  /* Calculate gradient for each Weight (error * corresponding input); In case of bias, the input is 1.0 */
  Synapse_iterator<>::iterate(net.neuron_array(neuron_index).input_weights(),[&](
    Index_synapse_interval weight_synapse, sint32 weight_index
  ){
    addition = (
      get_derivative_for(solve_thread_index, sequence_index, weight_index)
      * *error_values[solve_thread_index][neuron_index]
    );
    buffer = *get_weight_gradient()[weight_index];
    while( /* try to add the calculated gradient to the accumulated value */
      !get_weight_gradient()[weight_index]->compare_exchange_weak( buffer, buffer + addition )
    )buffer = *get_weight_gradient()[weight_index];
  });
}

void Sparse_net_optimizer::normalize_weight_gradients(void){
  uint32 weight_index = 0;
  const uint32 weight_number = 1 + static_cast<uint32>(net.weight_table_size()/context.get_max_processing_threads());
  for( /* As long ast there are remaining threads to open and they would process existing Neurons */
    uint32 process_thread_index = 0;
    ( (process_thread_index < context.get_max_processing_threads())
      &&(static_cast<uint32>(net.weight_table_size()) > weight_index) );
    ++process_thread_index
  ){ /* Basicall for every calculated gradient.. */
      process_threads[0].push_back(thread(
        &Sparse_net_optimizer::normalize_weight_gradients_thread, this,
        weight_index, min(weight_number, (net.weight_table_size() - weight_index))
      ));
      weight_index += weight_number;
  }
  wait_for_threads(process_threads[0]);
}

void Sparse_net_optimizer::normalize_weight_gradients_thread(uint32 weight_index, uint32 weight_number){
  for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator)
    get_weight_gradient()[weight_index + weight_iterator]->store(
      *get_weight_gradient()[weight_index + weight_iterator] / 
      static_cast<sdouble32>(context.get_minibatch_size() * train_set.get_sequence_size())
    );
}

} /* namespace sparse_net_library */
