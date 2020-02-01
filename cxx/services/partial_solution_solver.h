#ifndef Partial_solution_H
#define Partial_solution_H

#include <vector>

#include "sparse_net_global.h"

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;
using std::reference_wrapper;

class Partial_solution_solver{

public:
  Partial_solution_solver(const Partial_solution& partial_solution, uint32 num_of_transition_data = 0)
  : detail(partial_solution)
  , internal_iterator(detail.get().inside_indices())
  , input_iterator(detail.get().input_data())
  , num_of_transitional_data(num_of_transition_data)
  , num_of_non_transitional_data(detail.get().internal_neuron_number() - num_of_transitional_data)
  , transfer_function_input(vector<sdouble32>(num_of_transition_data))
  , transfer_function_output(vector<sdouble32>(num_of_transition_data))
  , neuron_output(detail.get().internal_neuron_number())
  , collected_input_data(input_iterator.size())
  { reset(); }

  /**
   * @brief      Gets the size of the elements taken by the configurad Patial solution.
   *
   * @return     The input size in number of elements ( @sdouble32 ).
   */
  uint32 get_input_size(void) const;

  /**
   * @brief      Gets the raw input added into the transfer function, provided the @Partial_solution monitors for it
   *
   * @return     The array for the input values for the neurons.
   */
  vector<sdouble32> get_transfer_function_input(void) const{
     return transfer_function_input;
  }

  /**
   * @brief      Gets the output from to the transfer function, provided the @Partial_solution monitors for it
   *
   * @return     The array for the input values for the neurons.
   */
  vector<sdouble32> get_transfer_function_output(void) const{
    return transfer_function_output;
  }

  /**
   * @brief      Gives back the size of the array the required Gradient data is stored in.
   *             the gradient data contains intermediate calculations of the output layer neurons, 
   *             which is required to calcualte gradient information.
   *
   * @return     The gradient data array size.
   */
  const uint32 get_gradient_data_size() const{
    if(transfer_function_input.size() == transfer_function_output.size())
      return transfer_function_output.size();
    else throw "Neuron gradient information not consistent!";
  }

  /**
   * @brief      Collects the input stated inside the @Partial_solution into @collected_input_data
   *
   * @param      input_data   The input data
   * @param[in]  neuron_data  The neuron data
   */
  void collect_input_data(vector<sdouble32>& input_data, vector<sdouble32> neuron_data);

  /**
   * @brief      Solves the detail given in the argument, then cleans it up and returns the solution
   *
   * @return     The result data of the internal neurons
   */
  vector<sdouble32> solve();

  /**
   * @brief      Resets the data of the included Neurons.
   */
  void reset(void){
    for(sdouble32& neuron_data : neuron_output) neuron_data = 0;
  }

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a SparseNet
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_valid(void) const;

private:
  /**
   * The Partial solution to solve
   */
  reference_wrapper<const Partial_solution> detail;

  /**
   * The iterator to go through the Neurons while solving the detail
   */
  Synapse_iterator internal_iterator;

  /**
   * THe iterator to go through the inputs of the detail
   */
  Synapse_iterator input_iterator;

  /**
   * For Gradient information, intermeidate results are required to be stored.
   * The Partial solution solver shall store the last num_of_transitional_data 
   * of intermediate results in @transfer_function_input and @transfer_function_output
   */
  uint32 num_of_transitional_data;
  uint32 num_of_non_transitional_data;
  vector<sdouble32> transfer_function_input;
  vector<sdouble32> transfer_function_output;

  /**
   * The data collected from @Neurons when they are solved
   */
  vector<sdouble32> neuron_output;

  /**
   * The data collected from the @Partial_solution input
   */
  vector<sdouble32> collected_input_data;

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */
