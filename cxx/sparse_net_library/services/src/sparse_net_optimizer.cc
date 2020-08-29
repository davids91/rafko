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

#include "sparse_net_library/services/sparse_net_optimizer.h"

#include <atomic>
#include <thread>
#include <cmath>
#include <stdexcept>

#include "sparse_net_library/models/spike_function.h"

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;
using std::min;
using std::max;

Sparse_net_optimizer::Sparse_net_optimizer(
  SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
  shared_ptr<Cost_function> the_function, weight_updaters weight_updater_, Service_context& service_context
): net(neural_network)
,  context(service_context)
,  transfer_function(context)
,  net_solution(Solution_builder(context).build(net))
,  solvers()
,  train_set(train_set_)
,  test_set(test_set_)
,  set_mutex()
,  loops_unchecked(50)
,  sequence_truncation(min(context.get_memory_truncation(),train_set.get_sequence_size()))
,  gradient_step(Backpropagation_queue_wrapper(neural_network, context)())
,  cost_function(the_function)
,  solve_threads()
,  process_threads(context.get_max_solve_threads()) /* One queue for every solve thread */
,  neuron_data_sequences()
,  transfer_function_input(context.get_max_solve_threads())
,  error_values(context.get_max_solve_threads())
,  weight_derivatives(context.get_max_solve_threads())
,  weight_gradient()
{
  (void)context.set_minibatch_size(max(1u,min(
    train_set.get_number_of_sequences(),context.get_minibatch_size()
  )));
  solve_threads.reserve(context.get_max_solve_threads());
  for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
    solvers.push_back(make_unique<Solution_solver>(*net_solution, service_context));
    
    if(train_set.get_feature_size() != solvers.back()->get_output_size())
      throw std::runtime_error("Network output size doesn't match size of provided training labels!");

    if(test_set.get_feature_size() != solvers.back()->get_output_size())
      throw std::runtime_error("Network output size doesn't match size of provided testing labels!");

    neuron_data_sequences.push_back(Data_ringbuffer(train_set_.get_sequence_size(), neural_network.neuron_array_size()));
    error_values[threads] = vector<unique_ptr<atomic<sdouble32>>>();
    error_values[threads].reserve(net.neuron_array_size());
    weight_derivatives[threads] = vector<vector<unique_ptr<atomic<sdouble32>>>>(sequence_truncation);
    transfer_function_input[threads] = vector<vector<sdouble32>>(train_set.get_sequence_size());
    for(sint32 i = 0; i < net.neuron_array_size(); ++i)
      error_values[threads].push_back(make_unique<atomic<sdouble32>>());
    for(uint32 sequence_index = 0; sequence_index < train_set.get_sequence_size(); ++sequence_index){
      transfer_function_input[threads][sequence_index] = vector<sdouble32>();
      transfer_function_input[threads][sequence_index].reserve(net.neuron_array_size());
      if(sequence_index < sequence_truncation){
        weight_derivatives[threads][sequence_index] = vector<unique_ptr<atomic<sdouble32>>>();
        for(sint32 i = 0; i < net.weight_table_size(); ++i)
          weight_derivatives[threads][sequence_index].push_back(make_unique<atomic<sdouble32>>());
      }
    }
    process_threads[threads].reserve(context.get_max_processing_threads());
  }
  weight_gradient.reserve(net.weight_table_size());
  for(sint32 i = 0; i < net.weight_table_size(); ++i){
    weight_gradient.push_back(make_unique<atomic<sdouble32>>());
  }
  weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,context);
}

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
      thread_index < min(
        static_cast<uint32>(test_set.get_number_of_sequences()/samples_to_evaluate),
        static_cast<uint32>(context.get_max_solve_threads())
      );
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
  uint32 raw_label_start_index;
  for(uint32 sample_iterator = 0; sample_iterator < samples_to_evaluate; ++sample_iterator){
    uint32 inputs_index = (sample_start + sample_iterator) * (train_set.get_sequence_size() + train_set.get_prefill_inputs_number());
    for(uint32 prefill_iterator = 0; prefill_iterator < test_set.get_prefill_inputs_number(); ++prefill_iterator){
      solvers[solve_thread_index]->solve(test_set.get_input_sample(inputs_index));
      ++inputs_index;
    }
    raw_label_start_index = (sample_start + sample_iterator);
    for(uint32 sequence_iterator = 0; sequence_iterator < train_set.get_sequence_size(); ++sequence_iterator){
      solvers[solve_thread_index]->solve(test_set.get_input_sample(inputs_index));
      ++inputs_index;
    }
    test_set.set_features_for_labels(
      solvers[solve_thread_index]->get_neuron_memory().get_whole_buffer(),
      raw_label_start_index, (test_set.get_sequence_size() * test_set.get_feature_size())
    ); /* Re-calculate error for the training set for this run */
    solvers[solve_thread_index]->reset();
  }
}

void Sparse_net_optimizer::step_thread(uint32 solve_thread_index, uint32 samples_to_evaluate){
  uint32 raw_sample_start_index;
  uint32 raw_sample_index;
  uint32 raw_inputs_index;

  for(uint32 sample = 0; sample < samples_to_evaluate; ++sample){
    raw_sample_index = rand()%(train_set.get_number_of_sequences()); /* decide on the index of a random sample */
    raw_inputs_index = raw_sample_index * (train_set.get_sequence_size() + train_set.get_prefill_inputs_number()); /* calculate the raw input arrays index */
    raw_sample_index *= train_set.get_sequence_size(); /* calculate the raw labels array index */
    raw_sample_start_index = raw_sample_index;

    /* Evaluate the current sequence step by step */
    solvers[solve_thread_index]->reset();
    for(uint32 prefill_iterator = 0; prefill_iterator < train_set.get_prefill_inputs_number(); ++prefill_iterator){
      solvers[solve_thread_index]->solve(train_set.get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
      ++raw_inputs_index;
    }
    for(uint32 sequence_iterator = 0; sequence_iterator < train_set.get_sequence_size(); ++sequence_iterator){
      neuron_data_sequences[solve_thread_index].step();
      solvers[solve_thread_index]->solve(train_set.get_input_sample(raw_inputs_index)); /* Solve the network for the sampled labels input */
      transfer_function_input[solve_thread_index][sequence_iterator] = solvers[solve_thread_index]->get_transfer_function_input();
      neuron_data_sequences[solve_thread_index].copy_latest(solvers[solve_thread_index]->get_neuron_memory());

      /* Only calculate the derivatives for the first un-truncated sequences */
      if(sequence_iterator < sequence_truncation){ /* Since the network will be the same, the derivatives can be re-used for the later sequences */
        for(unique_ptr<atomic<sdouble32>>& derivative_value : weight_derivatives[solve_thread_index][sequence_iterator]) *derivative_value = 0;
        calculate_derivatives(solve_thread_index, sequence_iterator, raw_inputs_index, raw_sample_index);
      }
      ++raw_sample_index;
      ++raw_inputs_index;
    }

    set_mutex.lock();
    train_set.set_features_for_labels(
      neuron_data_sequences[solve_thread_index].get_whole_buffer(),
      raw_sample_start_index, (train_set.get_sequence_size() * train_set.get_feature_size())
    ); /* Re-calculate error for the training set */
    set_mutex.unlock();

    /* Calculate the gradients from the current sequence */
    for(sint32 sequence_iterator = train_set.get_sequence_size()-1; sequence_iterator >= 0 ; --sequence_iterator){
      --raw_sample_index;
      --raw_inputs_index;

      for(unique_ptr<atomic<sdouble32>>& error_value : error_values[solve_thread_index])
        *error_value = 0;

      calculate_output_errors(solve_thread_index, sequence_iterator, raw_inputs_index, raw_sample_index);
      propagate_output_errors_back(solve_thread_index, sequence_iterator);
      accumulate_weight_gradients(solve_thread_index, sequence_iterator);
    }
    solvers[solve_thread_index]->reset();
    neuron_data_sequences[solve_thread_index].reset();
  }
}

void Sparse_net_optimizer::calculate_derivatives(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index){
  uint32 neuron_index = net.neuron_array_size()-net.output_neuron_number(); /* Start from the output layer */
  const uint32 neuron_number = 1 + (net.output_neuron_number()/static_cast<uint32>(context.get_max_processing_threads()));
  for( /* As long as there are free thread-slots or remaining neurons to be processed.. */
    uint32 process_thread_index = 0;
    process_thread_index < min(static_cast<uint32>(context.get_max_processing_threads()),static_cast<uint32>(net.output_neuron_number()));
    ++process_thread_index
  ){ /* Iterate through the neurons in the network */
    process_threads[solve_thread_index].push_back(thread(
      &Sparse_net_optimizer::calculate_derivatives_thread, this, solve_thread_index, sequence_index, raw_inputs_index, raw_sample_index,
      neuron_index, min(neuron_number, (net.output_neuron_number() - neuron_index))
    ));
    neuron_index += neuron_number;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::calculate_derivatives_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index, uint32 neuron_index, uint32 neuron_number){
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
          addition = train_set.get_input_sample(raw_inputs_index)[Synapse_iterator<>::input_index_from_synapse_index(
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

void Sparse_net_optimizer::calculate_output_errors(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index){
  uint32 neuron_index = net.neuron_array_size()-net.output_neuron_number(); /* Start from the output layer */
  const uint32 neuron_number = 1 + (net.output_neuron_number()/static_cast<uint32>(context.get_max_processing_threads()));
  for( /* As long as there are free thread-slots or remaining neurons to be processed.. */
    uint32 process_thread_index = 0;
    process_thread_index < min(static_cast<uint32>(context.get_max_processing_threads()),static_cast<uint32>(net.output_neuron_number()));
    ++process_thread_index
  ){ /* Iterate through the neurons in the network */
    process_threads[solve_thread_index].push_back(thread(
      &Sparse_net_optimizer::calculate_output_errors_thread, this, solve_thread_index, sequence_index, raw_inputs_index, raw_sample_index,
      neuron_index, min(neuron_number, (net.output_neuron_number() - neuron_index))
    ));
    neuron_index += neuron_number;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::calculate_output_errors_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 raw_inputs_index, uint32 raw_sample_index, uint32 neuron_index, uint32 neuron_number){
  sdouble32 buffer;
  sdouble32 addition;
  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    addition = cost_function->get_d_cost_over_d_feature(
      ((neuron_index + neuron_iterator) - (net.neuron_array_size() - net.output_neuron_number())),
      train_set.get_label_sample(raw_sample_index),
      neuron_data_sequences[solve_thread_index].get_const_element(sequence_index,Input_synapse_interval()),
      train_set.get_number_of_sequences()
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

void Sparse_net_optimizer::accumulate_weight_gradients(uint32 solve_thread_index, uint32 sequence_index){
  const uint32 neurons_to_process = 1 + (net.neuron_array_size()/context.get_max_processing_threads());
  uint32 neuron_index = 0;
  for( /* As long as there are threads to open or remaining neurons */
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_processing_threads())
      &&(static_cast<uint32>(net.neuron_array_size()) > neuron_index) );
    ++thread_index
  ){
    process_threads[solve_thread_index].push_back(thread(
      &Sparse_net_optimizer::accumulate_weight_gradients_thread, this,
      solve_thread_index, sequence_index, neuron_index,
      min(neurons_to_process, static_cast<uint32>(net.neuron_array_size() - neuron_index))
    ));
    neuron_index += neurons_to_process;
  }
  wait_for_threads(process_threads[solve_thread_index]);
}

void Sparse_net_optimizer::accumulate_weight_gradients_thread(
  uint32 solve_thread_index, uint32 sequence_index, uint32 neuron_index, uint32 neuron_number
){
  sdouble32 buffer;
  sdouble32 addition;

  for(uint32 neuron_iterator = neuron_index; neuron_iterator < (neuron_index + neuron_number); ++neuron_iterator){
    /* Calculate gradient for each Weight (error * corresponding input); In case of bias, the input is 1.0 */
    Synapse_iterator<>::iterate(net.neuron_array(neuron_iterator).input_weights(),[&](
      Index_synapse_interval weight_synapse, sint32 weight_index
    ){
      addition = (
        get_derivative_for(solve_thread_index, sequence_index, weight_index)
        * *error_values[solve_thread_index][neuron_iterator]
      );
      buffer = *get_weight_gradient()[weight_index];
      while( /* try to add the calculated gradient to the accumulated value */
        !get_weight_gradient()[weight_index]->compare_exchange_weak( buffer, buffer + addition )
      )buffer = *get_weight_gradient()[weight_index];
    });
  }
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
