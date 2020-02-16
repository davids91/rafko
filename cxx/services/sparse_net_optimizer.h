#ifndef SPARSE_NET_OPTIMIZER_H
#define SPARSE_NET_OPTIMIZER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/cost_function.h"
#include "models/transfer_function.h"
#include "services/solution_builder.h"
#include "services/solution_solver.h"
#include "services/function_factory.h"
#include "services/backpropagation_queue_wrapper.h"


#include <vector>
#include <memory>

namespace sparse_net_library{

using std::vector;
using std::unique_ptr;
using google::protobuf::Arena;

class Sparse_net_optimizer{
public:
  Sparse_net_optimizer(
    SparseNet& neural_network, vector<vector<sdouble32>>& label_samples_,
    Service_context service_context = Service_context()
  ): context(service_context)
  ,  label_samples(label_samples_)
  ,  net(neural_network)
  ,  transfer_function(context)
  ,  net_solution(*Solution_builder().service_context(context).build(net))
  ,  solver(context.get_max_solve_threads(), Solution_solver(net_solution, service_context))
  ,  gradient_step(Backpropagation_queue_wrapper(neural_network)())
  ,  cost_function(Function_factory::build_cost_function(net,label_samples))
  ,  neuron_router(net)
  ,  error_values(context.get_max_solve_threads())
  ,  weight_gradients(0)
  ,  feature_buffers(context.get_max_solve_threads())
  , step_size(1e-3)
  {
    for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads)
      for(sint32 i = 0; i < net.neuron_array_size(); ++i)
        error_values[threads].push_back(std::make_unique<atomic<sdouble32>>());
    for(sint32 i = 0; i < net.weight_table_size(); ++i)
      weight_gradients.push_back(std::make_unique<atomic<sdouble32>>());
  };

   /**
    * @brief      Step the net in the opposite direction of the gradient slope 
    *
    * @param      input_samples  The input samples to base the error function on
    * @param[in]  sample_size    The number of feature-label pairs considered to be 1 sample
    *                            this is important for recurrent Networks, ans the sample size 
    *                            shall set how deep shall the Back propagation through time go,
    *                            and it also considers @smaple_size number of samples in the array 
    *                            as 1 actual sample.
    */
  void step(
    vector<vector<sdouble32>>& input_samples, uint32 batch_size,
    sdouble32 step_size_ = 0, uint32 sequence_size = 1
  );

  /**
   * @brief      Gives back the error of the configured Network based on the previous optimization step
   *
   * @return     Error value
   */
  sdouble32 get_last_error(){
   return last_error;
  }

private:
  Service_context context;
  vector<vector<sdouble32>>& label_samples;
  SparseNet& net;
  Transfer_function transfer_function;

  Solution net_solution;
  vector<Solution_solver> solver; 

  Backpropagation_queue gradient_step;
  unique_ptr<Cost_function> cost_function;
  Neuron_router neuron_router;

  vector<vector<unique_ptr<atomic<sdouble32>>>> error_values;
  vector<unique_ptr<atomic<sdouble32>>> weight_gradients;
  vector<sdouble32> gradient_values;
  vector<vector<sdouble32>> feature_buffers;
  sdouble32 step_size;
  atomic<sdouble32> last_error;

  void calculate_gradient(vector<sdouble32>& input_sample, uint32 sample_index, uint32 solve_thread_index);

  void calculate_weight_gradients(vector<sdouble32>& input_sample, uint32 neuron_index, uint32 solve_thread_index);

  void update_weights_with_gradients(uint32 weight_index){
    net.set_weight_table( weight_index,
      net.weight_table(weight_index) + *weight_gradients[weight_index] * step_size
    );
  }

  void copy_weight_to_solution(
    uint32 inner_neuron_index,
    Partial_solution& partial,
    uint32 neuron_weight_synapse_starts,
    uint32 inner_neuron_weight_index_starts 
  );
  
  void propagate_errors_back(uint32 neuron_index, uint32 solve_thread_index){
    sdouble32 buffer;
    const Neuron& neuron = net.neuron_array(neuron_index);
    uint32 weight_index = 0;
    uint32 weight_synapse_index = 0;
    neuron_router.run_for_neuron_inputs(neuron_index,[&](sint32 child_index){
      if(!Synapse_iterator::is_index_input(child_index)){
        buffer = *error_values[solve_thread_index][child_index];
        while(!error_values[solve_thread_index][child_index]->compare_exchange_weak(
          buffer, 
          (buffer + *error_values[solve_thread_index][neuron_index] 
            * net.weight_table(neuron.input_weights(weight_synapse_index).starts() + weight_index))
        ))buffer = *error_values[solve_thread_index][child_index];
      }
      ++weight_index; 
      if(weight_index >= neuron.input_weights(weight_synapse_index).interval_size()){
        weight_index = 0; 
        ++weight_synapse_index;
      }
    });
  }
};

} /* namespace sparse_net_library */

#endif /* SPARSE_NET_OPTIMIZER_H */
