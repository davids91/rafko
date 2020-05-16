#ifndef SPARSE_NET_APPROXIMIZER_H
#define SPARSE_NET_APPROXIMIZER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "gen/training.pb.h"
#include "models/cost_function.h"
#include "models/data_aggregate.h"
#include "services/solution_builder.h"
#include "services/solution_solver.h"
#include "services/updater_factory.h"

#include <cmath>

namespace sparse_net_library{

using std::unique_ptr;
using std::make_unique;
using std::max;
using std::min;

/**
 * @brief      This class approximates gradients for a @Dataset and @Sparse_net.
 *             The approximated gradients are collected into one gradient fragment.
 */
class Sparse_net_approximizer{
public:
    Sparse_net_approximizer(
    SparseNet& neural_network, Data_aggregate& train_set_, Data_aggregate& test_set_,
    weight_updaters weight_updater_, Service_context service_context = Service_context()
  ): net(neural_network)
  ,  context(service_context)
  ,  net_solution(Solution_builder().service_context(context).build(net))
  ,  solvers()
  ,  train_set(train_set_)
  ,  test_set(test_set_)
  ,  gradient_fragment()
  ,  loops_unchecked(50)
  ,  sequence_truncation(min(context.get_memory_truncation(),train_set.get_sequence_size()))
  ,  cost_function(Function_factory::build_cost_function(net, train_set.get_number_of_samples(), context))
  ,  solve_threads()
  ,  process_threads(context.get_max_solve_threads()) /* One queue for every solve thread */
  /* Cache variables */
  ,  tmp_synapse_interval()
  ,  initial_error(0)
  {
    (void)context.set_minibatch_size(max(1u,min(
      train_set.get_number_of_sequences(),context.get_minibatch_size()
    )));
    solve_threads.reserve(context.get_max_solve_threads());
    for(uint32 threads = 0; threads < context.get_max_solve_threads(); ++threads){
      solvers.push_back(make_unique<Solution_solver>(*net_solution, service_context));
      process_threads[threads].reserve(context.get_max_processing_threads());
    }
    weight_updater = Updater_factory::build_weight_updater(net,weight_updater_,context);
  };

  ~Sparse_net_approximizer(){ solvers.clear(); }
  Sparse_net_approximizer(const Sparse_net_approximizer& other) = delete;/* Copy constructor */
  Sparse_net_approximizer(Sparse_net_approximizer&& other) = delete; /* Move constructor */
  Sparse_net_approximizer& operator=(const Sparse_net_approximizer& other) = delete; /* Copy assignment */
  Sparse_net_approximizer& operator=(Sparse_net_approximizer&& other) = delete; /* Move assignment */

  /**
   * @brief      Step the net in the opposite direction of the gradient slope
   */
  void collect(void);

  /**
   * @brief      Applies the colleted gradient fragment to the configured network
   */
  void apply_fragment(void);

  /**
   * @brief      Gives back the error of the configured Network based on the training dataset
   */
  sdouble32 get_train_error() const{
   return train_set.get_error();
  }

  /**
   * @brief      Gives back the error of the configured Network based on the test set
   */
  sdouble32 get_test_error() const{
   return test_set.get_error();
  }

  /**
   * @brief      Helper function to get the collected weight gradient fragment
   *
   * @return     Constant reference to the current weight gradients array
   */
  const Gradient_fragment& get_weight_gradient(void) const{
    return gradient_fragment;
  }

private:
  SparseNet& net;
  Service_context context;
  unique_ptr<Solution> net_solution;
  vector<unique_ptr<Solution_solver>> solvers;
  Data_aggregate& train_set;
  Data_aggregate& test_set;
  Gradient_fragment gradient_fragment;

  uint32 loops_unchecked;
  uint32 sequence_truncation;
  unique_ptr<Cost_function> cost_function;
  unique_ptr<Weight_updater> weight_updater;

  vector<thread> solve_threads; /* The threads to be started during optimizing the network */
  vector<vector<thread>> process_threads; /* The inner process thread to be started during net optimization */

  Index_synapse_interval tmp_synapse_interval;
  sdouble32 initial_error;

  /**
   * @brief      A thread to approximate the gradient for a weight for a given number of samples.
   *
   * @param[in]  solve_thread_index   The index of the used thread
   * @param[in]  samples_to_evaluate  The number of samples to evaluate
   */
  void collect_thread(uint32 solve_thread_index, uint32 sequence_index, uint32 sequences_to_evaluate);

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

#endif /* SPARSE_NET_APPROXIMIZER_H */