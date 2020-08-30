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

#ifndef Partial_solution_H
#define Partial_solution_H

#include "sparse_net_global.h"

#include <vector>
#include <stdexcept>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/models/data_ringbuffer.h"
#include "sparse_net_library/services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;

class Partial_solution_solver{

public:
  Partial_solution_solver(const Partial_solution& partial_solution, Data_ringbuffer& neuron_data_, Service_context& service_context)
  :  detail(partial_solution)
  ,  neuron_data(neuron_data_)
  ,  internal_iterator(detail.weight_indices())
  ,  input_iterator(detail.input_data())
  ,  transfer_function_input(detail.internal_neuron_number(),0)
  ,  transfer_function_output(detail.internal_neuron_number(),0)
  ,  collected_input_data(input_iterator.size())
  ,  transfer_function(service_context)
  { 
    if(transfer_function_input.size() != transfer_function_output.size())
      throw std::runtime_error("Neuron gradient data Incompatible!");
    reset();
  }

  /**
   * @brief      Gets the size of the elements taken by the configurad Patial solution.
   *
   * @return     The input size in number of elements ( @sdouble32 ).
   */
  uint32 get_input_size(void) const{
    return collected_input_data.size();
  }

  /**
   * @brief      Collects the input stated inside the @Partial_solution into @collected_input_data
   *             from the data in @input_data and the neuron data provided by @solver
   *
   * @param      input_data   The input data given to the network
   */
  void collect_input_data(const vector<sdouble32>& input_data);

  /**
   * @brief      Provides the gradient data to the given references
   *
   * @param      transfer_function_input_   The reference to transfer function input
   * @param      transfer_function_output_  The reference to transfer function output
   */
  void provide_gradient_data(vector<sdouble32>& transfer_function_input_, vector<sdouble32>& transfer_function_output_) const;

  /**
   * @brief      Solves the partial solution in the given argument and updates the reference @neuron_data
   *             also produces helper data through @provide_gradient_data
   */
  void solve();

  /**
   * @brief      Resets the data of the included Neurons.
   */
  void reset(void){
    for(sdouble32& neuron_data : transfer_function_input) neuron_data = 0;
    for(sdouble32& neuron_data : transfer_function_output) neuron_data = 0;
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
  const Partial_solution& detail;

  /**
   * The reference from which the input can be collected, and output can be provided to
   */
  Data_ringbuffer& neuron_data;

  /**
   * The iterator to go through the Neuron weights while solving the detail
   */
  Synapse_iterator<> internal_iterator;

  /**
   * The iterator to go through the I/O of the detail
   */
  Synapse_iterator<Input_synapse_interval> input_iterator;

  /**
   * For Gradient information, intermeidate results are required to be stored.
   * The Partial solution solver shall store the some intermediate results here
   */
  vector<sdouble32> transfer_function_input;
  vector<sdouble32> transfer_function_output;

  /**
   * The data collected from the @Partial_solution input
   */
  vector<sdouble32> collected_input_data;

  /**
   * The transfer function set configured for the current session
   */
  Transfer_function transfer_function;

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */
