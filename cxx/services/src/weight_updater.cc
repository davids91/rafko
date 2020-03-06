#include "services/weight_updater.h"

#include "services/synapse_iterator.h"

namespace sparse_net_library{

void Weight_updater::update_weights_with_gradients(
  vector<unique_ptr<atomic<sdouble32>>>& gradients,
  vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
){
  uint32 weight_index = 0;
  const uint32 weight_number = 1 + static_cast<uint32>(net.weight_table_size()/context.get_max_solve_threads());
  for(
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_solve_threads())
      &&(static_cast<uint32>(net.weight_table_size()) >= weight_index) );
    ++thread_index
  ){ /* For every provided sample */
      calculate_threads.push_back(thread(
        &Weight_updater::update_weight_with_gradient, this, 
        weight_index, std::min(weight_number, (net.weight_table_size() - weight_index)),
        ref(gradients), ref(previous_gradients)
      ));
      weight_index += weight_number;
  }
  wait_for_threads(calculate_threads);
}

void Weight_updater::update_solution_with_weights(Solution& solution){
  uint32 process_thread_iterator = 0;
  uint32 neuron_weight_synapse_starts = 0;
  uint32 inner_neuron_weight_index_starts = 0;
  for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
    Partial_solution& partial = *solution.mutable_partial_solutions(partial_index);
    process_thread_iterator = 0;
    neuron_weight_synapse_starts = 0;
    inner_neuron_weight_index_starts = 0;
    while(
      (context.get_max_processing_threads() > calculate_threads.size())
      &&(process_thread_iterator < partial.internal_neuron_number())
    ){
      calculate_threads.push_back(thread(
        &Weight_updater::copy_weight_to_solution, this, process_thread_iterator,
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
    wait_for_threads(calculate_threads);
  } /* for(uint32 partial_index = 0;partial_index < solution.partial_solutions_size(); ++partial_index) */
}

void Weight_updater::copy_weight_to_solution(
uint32 inner_neuron_index, Partial_solution& partial,
uint32 neuron_weight_synapse_starts, uint32 inner_neuron_weight_index_starts 
){ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  uint32 weight_interval_index = 0;
  uint32 weight_synapse_index = 0;
  Neuron& neuron = *net.mutable_neuron_array(partial.actual_index(inner_neuron_index));
  partial.set_weight_table(partial.bias_index(inner_neuron_index),net.weight_table(neuron.bias_idx()));
  partial.set_weight_table(partial.memory_filter_index(inner_neuron_index),net.weight_table(neuron.memory_filter_idx()));
  weights_copied += 2;
  Synapse_iterator::iterate(net.neuron_array(partial.actual_index(inner_neuron_index)).input_indices(),[&](sint32 child_index){
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

} /* namespace sparse_net_library */