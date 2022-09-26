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

#ifndef PARTIAL_SOLUTION_SOLVER_H
#define PARTIAL_SOLUTION_SOLVER_H

#include "rafko_global.hpp"

#include <vector>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"

#include "rafko_utilities/models/data_pool.hpp"
#include "rafko_utilities/models/data_ringbuffer.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_net/services/synapse_iterator.hpp"

namespace rafko_net {

class RAFKO_EXPORT PartialSolutionSolver{

public:
  PartialSolutionSolver(const PartialSolution& partial_solution, const rafko_mainframe::RafkoSettings& settings)
  :  m_partialSolution(partial_solution)
  ,  m_internal_weight_iterator(m_partialSolution.weight_indices())
  ,  m_input_iterator(m_partialSolution.input_data())
  ,  m_transfer_function(settings)
  { }

  /**
   * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference;
   *             uses the common internal data pool for storing intermediate calculations
   *
   * @param      input_data           The reference to collect input data from
   * @param      output_neuron_data   The reference to transfer function output
   */
   void solve(const std::vector<double>& input_data, rafko_utilities::DataRingbuffer<>& output_neuron_data) const{
     std::vector<double>& used_buffer = m_commonDataPool.reserve_buffer(get_required_tmp_data_size());
     solve_internal(input_data, output_neuron_data, used_buffer);
     m_commonDataPool.release_buffer(used_buffer);
   }

   /**
    * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference;
    *             uses the provided data pool for storing intermediate calculations
    *
    * @param      input_data           The reference to collect input data from
    * @param      output_neuron_data   The reference to transfer function output
    * @param      used_data_pool       The reference to a datapool the partial solver may use for intermediate calculations
    */
   void solve(const std::vector<double>& input_data, rafko_utilities::DataRingbuffer<>& output_neuron_data, rafko_utilities::DataPool<double>& used_data_pool) const{
     std::vector<double>& used_buffer = used_data_pool.reserve_buffer(get_required_tmp_data_size());
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
   void solve(const std::vector<double>& input_data, rafko_utilities::DataRingbuffer<>& output_neuron_data, std::vector<double>& temp_data) const{
     temp_data.resize(get_required_tmp_data_size());
     solve_internal(input_data, output_neuron_data, temp_data);
   }

   /**
    * @brief      Provides the number of vector elements needed to solve the stored partial solution to
    *             store the temporary data for the calculations
    *
    * @return     True if detail is valid, False otherwise.
    */
   std::uint32_t get_required_tmp_data_size() const{
     return m_input_iterator.size();
   }

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a RafkoNet
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_valid() const;

private:
  static rafko_utilities::DataPool<double> m_commonDataPool;
  const PartialSolution& m_partialSolution;
  SynapseIterator<> m_internal_weight_iterator;
  SynapseIterator<InputSynapseInterval> m_input_iterator;
  TransferFunction m_transfer_function;

  /**
   * @brief      Solves the partial solution in the given argument and loads the result into a provided output reference
   *             and uses the provided vector for storing intermediate calculations. temp_data needs to be appropriately sized
   *             to ensure that there is enough elements in it to collect all required partial solution input data
   *
   * @param      input_data           The reference to collect input data from
   * @param      output_neuron_data   The reference to transfer function output
   * @param      temp_data            The reference a vector allocated to keep the required collected inputs
   */
  void solve_internal(const std::vector<double>& input_data, rafko_utilities::DataRingbuffer<>& output_neuron_data, std::vector<double>& temp_data) const;

};

} /* namespace rafko_net */
#endif /* PARTIAL_SOLUTION_SOLVER_H */
