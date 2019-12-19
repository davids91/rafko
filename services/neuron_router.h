#ifndef NEURON_ROUTER_H
#define NEURON_ROUTER_H

#include <deque>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"
namespace sparse_net_library {

using std::unique_ptr;
using std::vector;
using std::deque;
using std::atomic;

/**
 * @brief      This class describes a neuron router which iterates through the given @SparseNet,
               collecting a subset of Neurons from the thread, all of whom are able to be solved without
               waiting for any other Neurons. The subset is being collected based on the input relations
               between the Neurons. The Neurons at the beginning of the net only take in input data,
               so they already have their inputs ready. Any other Neurons build upon that, with each Iteration
               some additional @Neuron nodes are collected into a subset. That subset is later to be used by
               the @Solution_builder to compile @Partial_solutions.
               If a Neuron is solvable, its state is being set to "reserved", and collected into the subset.
               After an iteration the state update from the subset needs to be handled by whoever has access to
               the Neuron indexes inside.
 */
class Neuron_router{
public:
  Neuron_router(SparseNet& sparse_net);

  friend void swap_neuron_routers(Neuron_router& first, Neuron_router& second) noexcept;
  Neuron_router(const Neuron_router& other);
  Neuron_router(Neuron_router&& other);
  Neuron_router& operator=(Neuron_router other);
  Neuron_router& operator=(Neuron_router&& other);
  ~Neuron_router();

/**
 * @brief      Collects Neurons into a subset of the net
 *
 * @param[in]  iteration                 The iteration
 * @param[in]  arg_max_solve_threads     The argument maximum solve threads
 * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
 */
void collect_subset(uint16 iteration, uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes);

bool get_first_neuron_index_from_subset(uint32& put_it_here);
bool confirm_first_subset_element_processed(uint32 neuron_index);
void correct_subset_memory_usage(sdouble32 memory_delta);

  /**
   * @brief      Gives back Iteration state
   *
   * @return     true if the iteration of the net is finished
   */
  inline bool finished() const{
    return (
      (static_cast<int>(output_layer_iterator) == (net.neuron_array_size()-1))
      &&(is_neuron_processed(output_layer_iterator))
    );
  }
  inline bool is_neuron_in_progress(uint32 neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] > *neuron_states[neuron_index]);
  }
  inline bool is_neuron_reserved(uint32 neuron_index) const{
    return (neuron_state_reserved_value(neuron_index) == *neuron_states[neuron_index]);
  }
  inline bool is_neuron_solvable(uint32 neuron_index) const{
    return (neuron_number_of_inputs[neuron_index] == *neuron_states[neuron_index]);
  }
  inline bool is_neuron_processed(uint32 neuron_index) const{
    return (neuron_state_processed_value(neuron_index) == *neuron_states[neuron_index]);
  }
private:

  void set_neuron_to_processed(uint32 neuron_index);

  /**
   * @brief      Called form inside @collect_subset; A thread to handle @collect_subset
   *
   * @param[in]  iteration                 The iteration
   * @param[in]  arg_max_solve_threads     The argument maximum solve threads
   * @param[in]  arg_device_max_megabytes  The argument device maximum megabytes
   * @param[in]  thread_index              The thread index
   */
  void collect_subset_thread(uint16 iteration, uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes,uint8 thread_index);

  /**
   * @brief      Called form inside @collect_subset_thread; Checking the current Neuron and its input states
   *             updates its state accordingly
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @param      iteration      The number of times the algorithm ran to look for Neuron candidates, it is used to decide relevance to the currently finished subset
   * @return     The next neuron to move the iteration to
   */
  uint32 get_next_neuron(vector<uint32>& visiting, uint16& iteration);

  /**
   * @brief      Called form inside @collect_subset_thread; Adds a neuron into subset and updates relevant build states
   *
   * @param      net            The Sparse Net to be used
   * @param      visiting       A Vector containing the currently visiting Neuron along with the path leading to it
   * @param      visiting_next  The Next Neuron Candidate, which might be the same as the latest visit ( that means no candidates found to move to)
   * @param      iteration      The number of times the algorithm ran to look for Neuron candidates, it is used to decide relevance to the currently finished subset
   */
  void add_neuron_into_subset(vector<uint32>& visiting, uint32& visiting_next, uint16& iteration);

  /**
   * @brief      Decides the next Neuron to iterate to and increases the output layer iterator if needed
   *
   * @param      visiting       The visiting
   * @param      visiting_next  The visiting next
   * @param      iteration      The iteration
   */
  void step(vector<uint32>& visiting, uint32& visiting_next, uint16& iteration);

  SparseNet& net;

  /**
   * Number of already processed output layer Neurons
   */
  atomic<uint32> output_layer_iterator;

  /**
   * For each @Neuron in @SparseNet stores the processed state. Values:
   *  - Number of processed children ( storing raw children number without synapse information )
   *  - Number of processed children + 1 in case the Neuron is reserved
   *  - Number of processed children + 2 in case the Neuron is processed
   */
  vector<unique_ptr<atomic<uint32>>> neuron_states;

  /**
   * Number of inputs a Neuron has, based on the input index synapse sizes
   */
  vector<uint32> neuron_number_of_inputs;

  /**
   * A subset of the net representing independent solutions
   */
  std::mutex net_subset_mutex;
  std::atomic<sdouble32> net_subset_size; /* The size of the currently partial solution to be built in bytes */
  deque<uint32> net_subset_index;
  deque<uint32> net_subset; /*!#4 add transitively dependent neurons if memory allows it */

  /**
   * @brief      Inline functions to help build partial solutions
   *
   * @param[in]  neuron_index  The neuron index inside @neuron_number_of_inputs and @neuron_states
   *
   * @return     Information depending on the function
   */
  inline uint32 neuron_state_reserved_value(uint32 neuron_index) const{
    return neuron_number_of_inputs[neuron_index] + 1u;
  }
  inline uint32 neuron_state_processed_value(uint32 neuron_index) const{
    return neuron_number_of_inputs[neuron_index] + 2u;
  }
  inline sint32 neuron_state_iteration_value(uint32 neuron_index) const{
    return (*neuron_states[neuron_index] - neuron_state_processed_value(neuron_index));
  }
  inline uint32 neuron_iteration_relevance(uint32 neuron_index) const{
    return static_cast<uint32>(std::max( 0, neuron_state_iteration_value(neuron_index) ));
  }
  inline sint32 neuron_state_next_iteration_value(uint32 neuron_index, uint16 iteration) const{
    return (neuron_state_processed_value(neuron_index) + iteration + 1u);
  }
  inline bool is_neuron_subset_candidate(uint32 neuron_index, uint16 iteration) const{
    return(
      (neuron_iteration_relevance(neuron_index) <= iteration)
      &&(!is_neuron_processed(neuron_index))
      &&(!is_neuron_reserved(neuron_index))
    );
  }
};

} /* namespace sparse_net_library */
#endif /* NEURON_ROUTER_H */
