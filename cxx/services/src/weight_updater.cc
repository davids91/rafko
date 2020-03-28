#include "services/weight_updater.h"

#include "services/synapse_iterator.h"

namespace sparse_net_library{

void Weight_updater::calculate_velocity(const vector<unique_ptr<atomic<sdouble32>>>& gradients){
  uint32 weight_index = 0;
  const uint32 weight_number = 1 + static_cast<uint32>(net.weight_table_size()/context.get_max_solve_threads());
  for( /* As long as there are threads to open or remaining weights */
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_solve_threads())
      &&(static_cast<uint32>(net.weight_table_size()) > weight_index) );
    ++thread_index
  ){
    calculate_threads.push_back(thread(
      &Weight_updater::calculate_velocity_thread, this, ref(gradients),
      weight_index, std::min(weight_number, (net.weight_table_size() - weight_index))
    ));
    weight_index += weight_number;
  }
  wait_for_threads(calculate_threads);
}

void Weight_updater::update_weights_with_velocity(){
  uint32 weight_index = 0;
  const uint32 weight_number = 1 + static_cast<uint32>(net.weight_table_size()/context.get_max_solve_threads());
  for( /* As long as there are threads to open or remaining weights */
    uint32 thread_index = 0; 
    ( (thread_index < context.get_max_solve_threads())
      &&(static_cast<uint32>(net.weight_table_size()) > weight_index) );
    ++thread_index
  ){
    calculate_threads.push_back(thread(
      &Weight_updater::update_weight_with_velocity, this,
      weight_index, std::min(weight_number, (net.weight_table_size() - weight_index))
    ));
    weight_index += weight_number;
  }
  wait_for_threads(calculate_threads);
}

void Weight_updater::update_solution_with_weights(Solution& solution){
  uint32 inner_neuron_iterator = 0;
  uint32 neuron_weight_synapse_starts = 0;
  uint32 inner_neuron_weight_index_starts = 0;
  uint32 neuron_index;
  for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
    Synapse_iterator output_iterator(solution.partial_solutions(partial_index).output_data());
    inner_neuron_iterator = 0;
    neuron_weight_synapse_starts = 0;
    inner_neuron_weight_index_starts = 0;
    while(inner_neuron_iterator < solution.partial_solutions(partial_index).internal_neuron_number()){
      while( /* As long as there are threads to open or remaining neurons */
        (context.get_max_processing_threads() > calculate_threads.size())
        &&(inner_neuron_iterator < solution.partial_solutions(partial_index).internal_neuron_number())
      ){
        neuron_index = output_iterator[inner_neuron_iterator];
        calculate_threads.push_back(thread(
          &Weight_updater::copy_weight_to_solution, this, neuron_index, inner_neuron_iterator,
          ref(*solution.mutable_partial_solutions(partial_index)), inner_neuron_weight_index_starts
        ));
        ++inner_neuron_weight_index_starts; /* ++ for memory filter */
        for(uint32 i = 0; i < solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_iterator); ++i){
          inner_neuron_weight_index_starts += 
            solution.partial_solutions(partial_index).weight_indices(neuron_weight_synapse_starts + i).interval_size();
        }
        neuron_weight_synapse_starts += solution.partial_solutions(partial_index).weight_synapse_number(inner_neuron_iterator);
        ++inner_neuron_iterator;
      }
      wait_for_threads(calculate_threads);
    }
  } /* for(uint32 partial_index = 0;partial_index < solution.partial_solutions_size(); ++partial_index) */
}

void Weight_updater::copy_weight_to_solution(
  uint32 neuron_index, uint32 inner_neuron_index, Partial_solution& partial, uint32 inner_neuron_weight_index_starts 
){ /*!Note: After shared weight optimization, this part is to be re-worked */
  uint32 weights_copied = 0;
  partial.set_weight_table(
    partial.memory_filter_index(inner_neuron_index),
    net.weight_table(net.neuron_array(neuron_index).memory_filter_idx())
  );
  ++weights_copied; /* ++ for Memory ratio */
  Synapse_iterator::iterate(net.neuron_array(neuron_index).input_weights(),[&](sint32 network_weight_index){
    partial.set_weight_table(
      (inner_neuron_weight_index_starts + weights_copied), net.weight_table(network_weight_index)
    );
    ++weights_copied;
  });
}

} /* namespace sparse_net_library */