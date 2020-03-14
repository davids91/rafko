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

#ifndef sparse_net_TEST_MOCKUPS_H
#define sparse_net_TEST_MOCKUPS_H

#include <vector>

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"

namespace sparse_net_library_test {

using std::vector;

using sparse_net_library::uint32;
using sparse_net_library::sdouble32;
using sparse_net_library::sdouble32;
using sparse_net_library::Partial_solution;
using sparse_net_library::SparseNet;

/**
 * @brief      generates a partial partial_solution manually based on the Neural Network structure:
 *             2 Neurons: The first neuron has the inputs and the second has the first neuron
 *
 * @param      partial_solution  The partial solution
 * @param[in]  number_of_inputs  The number of inputs to the network
 */
extern void manual_2_neuron_partial_solution(Partial_solution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset = 0);

/** @brief Calculates the result of the partial partial_solution manually based on the structure provided by @manual_2_neuron_partial_solution
 *         In case there are more than 2 inputs, all of them shall be processed with the same weight
 *         The end result shall be calculated as follows:
 *         - result1 = f[0]((input1 * weight1 + input2 * weight2) + bias1) //weighted inputs + biaf, given to the transfer function
 *         - result1 = (prev_result1 * memory_filter1) + (result1 * (double_literal(1.0)-memory_rato1) //apply memory filter
 *         - result2 = f[1]((result1 * weight3) + bias2)
 *         - end_result = (prev_result2 * memory_filter2) + (result2 * (double_literal(1.0)-memory_rato2)
 * @param[in]  partial_inputs      The collected inputs of the @Partial_solution
 * @param      prev_neuron_output  The previous neuron data
 * @param[in]  partial_solution    The partial solution containing the weights andmisc parameters of the calculation
 * @param[in]  neuron_offset       THe neuron offset in which the partial solution is present in the whole network
 */
extern void manual_2_neuron_result(const vector<sdouble32>& partial_inputs, vector<sdouble32>& prev_neuron_output, const Partial_solution& partial_solution, uint32 neuron_offset = 0);

/**
 * @brief      Calculates the result of a fully connected Network for the given inputs
 *
 * @param[in]  inputs           The inputs
 * @param[in]  neuron_data      The Neural data available to the network
 * @param[in]  layer_structure  The layer structure
 * @param[in]  network          The network
 */
extern void manaual_fully_connected_network_result(vector<sdouble32> inputs, vector<sdouble32>& neuron_data,
    vector<uint32> layer_structure, SparseNet network);

};

#endif /* sparse_net_TEST_MOCKUPS_H */
