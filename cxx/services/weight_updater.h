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
    SparseNet& sparse_net, Service_context service_context = Service_context(), uint32 required_iterations_for_step_ = 1
  ): net(sparse_net)
  ,  context(service_context)
  ,  required_iterations_for_step(required_iterations_for_step_)
  ,  iteration(0)
  ,  finished(false)
  ,  calculate_threads(0)
  { calculate_threads.reserve(context.get_max_processing_threads()); };

  /**
   * @brief      Do an iteration of weight updates. An actual weight update 
   *             shall count as valid when @required_iterations_for_step taken place.
   *
   * @param      gradients           The gradients
   * @param      previous_gradients  The previous gradients
   * @param      solution            The solution
   */
  void iterate(
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients,
    Solution& solution
  ){
    update_weights_with_gradients(gradients, previous_gradients);
    update_solution_with_weights(solution);
    iteration = (iteration + 1) % required_iterations_for_step;
    finished = (0 == iteration);
  }

  /**
   * @brief      The function to signal the weight updater that an iteration have started
   */
  void start(void){
    iteration = 0;
    finished = false;
  }

  /**
   * @brief      Tells if an iteration is at its valid state or not based on 
   *             he number of iterations since calling @start
   *
   * @return     True if finished, False otherwise.
   */
  bool is_finished(){
    return finished;
  }
  virtual ~Weight_updater() = default;

protected:
  SparseNet& net;
  Service_context context;
  const uint32 required_iterations_for_step;
  uint32 iteration;
  bool finished;
  
  /**
   * @brief      Gets the new value for one weight based on the gradients.
   *             More complex weight updaters should overwirte this function,
   *             as it is the basis of updating weights.
   *
   * @param[in]  weight_index        The weight index
   * @param      gradients           The gradients
   * @param      previous_gradients  The previous gradients
   *
   * @return     The new weight.
   */
  sdouble32 get_new_weight(
    uint32 weight_index,
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
  ){
    return(net.weight_table(weight_index) - (*gradients[weight_index] * context.get_step_size()));
  }

private:
  vector<thread> calculate_threads;

  /**
   * @brief      The function to update every weight of the referenced @SparseNet
   *             based on the values provided by @get_new_weight.
   *             It starts multiple threads, dividing almost equally the number of weights
   *             to be updated in each thread.
   *
   * @param      gradients           The gradients
   * @param      previous_gradients  The previous gradients
   */
  void update_weights_with_gradients(
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
  );

  /**
   * @brief      Copies the referenced @SparseNet weights into the solution in the arguments
   *             It supposes that the solution is one already built, and it is built from
   *             the same @SparseNet referenced in the updater. Opens up a thread for every internal 
   *             neuron in the partial solution, up until a maximum of Service_context::get_max_processing_threads.
   *
   * @param      solution  The solution
   */
  void update_solution_with_weights(Solution& solution);

  /**
   * @brief      A thread to update the weights of the @SpraseNet, called by @update_weights_with_gradients
   *
   * @param[in]  weight_index        The weight index
   * @param[in]  weight_number       The weight number
   * @param      gradients           The gradients
   * @param      previous_gradients  The previous gradients
   */
  void update_weight_with_gradient(
    uint32 weight_index, uint32 weight_number,
    vector<unique_ptr<atomic<sdouble32>>>& gradients,
    vector<unique_ptr<atomic<sdouble32>>>& previous_gradients
  ){
    for(uint32 weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
      net.set_weight_table(
        weight_index + weight_iterator, 
        get_new_weight((weight_index + weight_iterator),ref(gradients),ref(previous_gradients))
      );
    }
  }

  /**
   * @brief      A thread to copy a weight synapse from the referenced @SparseNet
   *             into a partial solution.
   *
   * @param[in]  inner_neuron_index                The inner neuron index
   * @param      partial                           The partial
   * @param[in]  neuron_weight_synapse_starts      The neuron weight synapse starts
   * @param[in]  inner_neuron_weight_index_starts  The inner neuron weight index starts
   */
  void copy_weight_to_solution(
    uint32 neuron_index, uint32 inner_neuron_index, Partial_solution& partial,
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