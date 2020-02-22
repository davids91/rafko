#ifndef WEIGHT_UPDATER_H
#define WEIGHT_UPDATER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "services/neuron_router.h"
#include "services/sparse_net_optimizer.h"

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
class Weight_updater{
public:
  Weight_updater(
    SparseNet& sparse_net,vector<unique_ptr<atomic<sdouble32>>>& weight_gradients_,
    Service_context& service_context
  ): net(sparse_net)
  ,  context(service_context)
  ,  weight_gradients(weight_gradients_)
  ,  calculate_threads(0)
  { 
    calculate_threads.reserve(context.get_max_processing_threads());
  };

  void update_weights_with_gradients(void){
    uint32 process_thread_iterator = 0;
    while(static_cast<int>(process_thread_iterator) < net.weight_table_size()){
      while(
        (context.get_max_processing_threads() > calculate_threads.size())
        &&(net.weight_table_size() > static_cast<int>(process_thread_iterator))
      ){
        calculate_threads.push_back(
          thread(&Weight_updater::update_weight_with_gradient, this, process_thread_iterator)
        );
        ++process_thread_iterator;
      }
      wait_for_threads(calculate_threads);
    }
  }

  void update_solution_with_weights(Solution& solution){
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

private:
  SparseNet& net;
  Service_context& context;
  vector<unique_ptr<atomic<sdouble32>>>& weight_gradients;
  vector<thread> calculate_threads;

  void update_weight_with_gradient(uint32 weight_index){
    net.set_weight_table( weight_index,
      net.weight_table(weight_index) + *weight_gradients[weight_index] * context.get_step_size()
    );
  }

  void copy_weight_to_solution(
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
    Neuron_router::run_for_neuron_inputs(net, partial.actual_index(inner_neuron_index),[&](sint32 child_index){
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

  /**
   * @brief      This function waits for the given threads to finish, ensures that every thread
   *             in the reference vector is finished, before it does.
   *
   * @param      calculate_threads  The calculate threads
   *//*!TODO: Find a better solution for these snippets */
  static void wait_for_threads(vector<thread>& calculate_threads){
    while(0 < calculate_threads.size()){
      if(calculate_threads.back().joinable()){
        calculate_threads.back().join();
        calculate_threads.pop_back();
      }
    }
  }
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_H */