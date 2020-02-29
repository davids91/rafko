#ifndef WEIGHT_UPDATER_H
#define WEIGHT_UPDATER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/service_context.h"

#include <thread>
#include <atomic>
#include <vector>

namespace sparse_net_library{

using std::ref;
using std::vector;
using std::atomic;
using std::thread;
using std::unique_ptr;

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

  void update_weights_with_gradients(void);

  void update_solution_with_weights(Solution& solution);

protected:
  SparseNet& net;
  Service_context& context;
  vector<unique_ptr<atomic<sdouble32>>>& weight_gradients;
  
  sdouble32 get_new_weight(uint32 weight_index){
    return(net.weight_table(weight_index) + (*weight_gradients[weight_index] * context.get_step_size()));
  }

private:
  vector<thread> calculate_threads;

  void update_weight_with_gradient(uint32 weight_index, uint32 weight_number){
    for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
      net.set_weight_table( weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator));
    }
  }

  void copy_weight_to_solution(
    uint32 inner_neuron_index, Partial_solution& partial,
    uint32 neuron_weight_synapse_starts, uint32 inner_neuron_weight_index_starts 
  );

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