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

#include "rafko_global.h"

#include <vector>
#include <atomic>
#include <mutex>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

#include "rafko_utilities/models/data_pool.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;
using std::mutex;
using std::atomic;

using rafko_utilities::DataPool;
using rafko_utilities::DataRingbuffer;

class Partial_solution_solver{

public:
  Partial_solution_solver(const Partial_solution& partial_solution, Service_context& service_context)
  :  detail(partial_solution)
  ,  internal_weight_iterator(detail.weight_indices())
  ,  input_iterator(detail.input_data())
  ,  transfer_function(service_context)
  { }

  /**
   * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference;
   *             uses the common internal data pool for storing intermediate calculations
   *
   * @param      input_data           The reference to collect input data from
   * @param      output_neuron_data   The reference to transfer function output
   */
   void solve(const vector<sdouble32>& input_data, DataRingbuffer& output_neuron_data) const{
     vector<sdouble32>& used_buffer = common_data_pool.reserve_buffer(get_required_tmp_data_size());
     solve_internal(input_data, output_neuron_data, used_buffer);
     common_data_pool.release_buffer(used_buffer);
   }

   /**
    * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference;
    *             uses the provided data pool for storing intermediate calculations
    *
    * @param      input_data           The reference to collect input data from
    * @param      output_neuron_data   The reference to transfer function output
    * @param      used_data_pool       The reference to a datapool the partial solver may use for intermediate calculations
    */
   void solve(const vector<sdouble32>& input_data, DataRingbuffer& output_neuron_data, DataPool<sdouble32>& used_data_pool) const{
     vector<sdouble32>& used_buffer = used_data_pool.reserve_buffer(get_required_tmp_data_size());
     solve_internal(input_data, output_neuron_data, used_buffer);
     used_data_pool.release_buffer(used_buffer);
   }

   /**
    * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference
    *             and uses the provided vector for storing intermediate calculations. It shall resize the provided temp_data
    *             to fit buffer needs.
    *
    * @param      input_data           The reference to collect input data from
    * @param      output_neuron_data   The reference to transfer function output
    * @param      temp_data            The reference a vector allocated to keep the required collected inputs
    */
   void solve(const vector<sdouble32>& input_data, DataRingbuffer& output_neuron_data,  vector<sdouble32>& temp_data) const{
     temp_data.resize(get_required_tmp_data_size());
     solve_internal(input_data, output_neuron_data, temp_data);
   }

   /**
    * @brief      Provides the number of vector elements needed to solve the stored partial solution to
    *             store the temporary data for the calculations
    *
    * @return     True if detail is valid, False otherwise.
    */
   uint32 get_required_tmp_data_size(void) const{
     return input_iterator.size();
   }

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a SparseNet
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_valid(void) const;

private:
  static DataPool<sdouble32> common_data_pool;

  /**
   * The Partial solution to solve
   */
  const Partial_solution& detail;

  /**
   * The iterator to go through the Neuron weights while solving the detail
   */
  Synapse_iterator<> internal_weight_iterator;

  /**
   * The iterator to go through the I/O of the detail
   */
  Synapse_iterator<Input_synapse_interval> input_iterator;

  /**
   * The transfer function set configured for the current session
   */
  Transfer_function transfer_function;

  /**
   * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference
   *             and uses the provided vector for storing intermediate calculations. temp_data needs to be appropriately sized
   *             to ensure that there is enough elements in it to collect all required partial solution input data
   *
   * @param      input_data           The reference to collect input data from
   * @param      output_neuron_data   The reference to transfer function output
   * @param      temp_data            The reference a vector allocated to keep the required collected inputs
   */
  void solve_internal(const vector<sdouble32>& input_data, DataRingbuffer& output_neuron_data, vector<sdouble32>& temp_data) const;

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */
