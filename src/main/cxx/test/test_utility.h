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

#ifndef TEST_UTILITY_H
#define TEST_UTILITY_H

#include <vector>
#include <memory>

#include "rafko_global.h"

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"

namespace rafko_test {

/**
 * @brief      generates a partial partial_solution manually based on the Neural Network structure:
 *             2 Neurons: The first neuron has the inputs and the second has the first neuron
 *
 * @param      partial_solution  The partial solution
 * @param[in]  number_of_inputs  The number of inputs to the network
 */
extern void manual_2_neuron_partial_solution(rafko_net::PartialSolution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset = 0);

/** @brief Calculates the result of the partial partial_solution manually based on the structure provided by @manual_2_neuron_partial_solution
 *         In case there are more than 2 inputs, all of them shall be processed with the same weight
 *         The end result shall be calculated as follows:
 *         - result1 = f[0]((input1 * weight1 + input2 * weight2) + bias1) //weighted inputs + bias, given to the transfer function
 *         - result1 = (prev_result1 * memory_filter1) + (result1 * (double_literal(1.0)-memory_rato1) //apply memory filter
 *         - result2 = f[1]((result1 * weight3) + bias2)
 *         - end_result = (prev_result2 * memory_filter2) + (result2 * (double_literal(1.0)-memory_rato2)
 * @param[in]  partial_inputs      The collected inputs of the @PartialSolution
 * @param      prev_neuron_output  The previous neuron data
 * @param[in]  partial_solution    The partial solution containing the weights andmisc parameters of the calculation
 * @param[in]  neuron_offset       THe neuron offset in which the partial solution is present in the whole network
 */
extern void manual_2_neuron_result(const std::vector<sdouble32>& partial_inputs, std::vector<sdouble32>& prev_neuron_output, const rafko_net::PartialSolution& partial_solution, uint32 neuron_offset = 0);

/**
 * @brief      Calculates the result of a fully connected Network for the given inputs
 *
 * @param[in]  inputs           The inputs
 * @param      previous_data    The data the Network should recieve from the previous run
 * @param[in]  neuron_data      The data the Neural network should contain in the current run
 * @param[in]  layer_structure  The layer structure
 * @param[in]  network          The network
 */
extern void manaual_fully_connected_network_result(
  std::vector<sdouble32>& inputs, std::vector<sdouble32> previous_data, std::vector<sdouble32>& neuron_data,
  std::vector<uint32> layer_structure, rafko_net::RafkoNet network
);

/**
 * @brief      Checks if the inputs are pointing to the same data and weight values are matching in the given
 *             @RafkoNet and @Solution
 *
 * @param      net       The net
 * @param      solution  The solution
 */
extern void check_if_the_same(rafko_net::RafkoNet& net, rafko_net::Solution& solution);

/**
 * @brief      Prints weights for the given arguments
 *
 * @param      net       The net
 * @param      solution  The solution
 */
extern void print_weights(rafko_net::RafkoNet& net, rafko_net::Solution& solution);

/**
 * @brief      Prints a training sample of the given data set, under the given index. Expects 2 inputs and one output!
 *
 * @param[in]  sample_sequence_index  The sample sequence index
 * @param      data_set               The data set
 * @param      net                    The net
 * @param      settings               The service settings
 */
extern void print_training_sample(uint32 sample_sequence_index, rafko_gym::RafkoDatasetWrapper& data_set, rafko_net::RafkoNet& net, rafko_mainframe::RafkoSettings& settings);

/**
 * @brief      Creates a normalized dataset for addition: basically adding two numbers together.
 *             The generated dataset is adequate for testing non-recurrent neural networks
 *
 * @param[in]  number_of_samples  The number of samples to create
 *
 * @return     The addition dataset. For each sample: Inputs: [a][b]; Outputs: [a+b]
 */
extern std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> create_addition_dataset(uint32 number_of_samples);

/**
 * @brief      Creates a normalized dataset for adding binary numbers: each number is stored
 *             as sequences of 0/1 one after another.
 *             The generated dataset is adequate for testing recurrent neural networks
 *
 * @param[in]  number_of_samples  The number of samples to create
 * @param[in]  sequence_size      The number of sequences one sample should contain, i.e. the size of the binary number
 *
 * @return     The addition dataset. For each sample: Inputs: [[a0][...][an]][[b0][...][bn]]; Labels: [[result0][...][resultn]]
 */
extern std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> create_sequenced_addition_dataset(uint32 number_of_samples, uint32 sequence_size);

/**
 * @brief      Checks if the two arguments match or not
 *
 * @param      sample_data      The sample data
 * @param      ringbuffer_data  The ringbuffer data
 */
extern void check_data_match(std::vector<sdouble32>& sample_data, std::vector<sdouble32>& ringbuffer_data);

/**
 * @brief      Generates a random Fully connected Dense network with some Layers set to have the softmax feature
 *
 * @param[in]  input_size       The size of the input vector accepted by the produces network
 * @param      settings         Contextual information
 *
 * @return     the generated network to be owned by the caller
 */
extern rafko_net::RafkoNet* generate_random_net_with_softmax_features(uint32 input_size, rafko_mainframe::RafkoSettings& settings);

/**
 * @brief      Creates a random dataset based on the given parameters
 *
 * @param[in]  input_size         The number of expected inputs are stored for the @DataSet
 * @param[in]  feature_size       The number of features and labels to be stored inside the @DataSet
 * @param[in]  sample_number      The number of samples to generate
 * @param[in]  sequence_size      The number of sequences one sample should contain
 * @param[in]  expected_label     The content of the dataset repeated in the correct structure
 *
 * @return     The created dataset
 */
extern std::unique_ptr<rafko_gym::DataSet> create_dataset(uint32 input_size, uint32 feature_size, uint32 sample_number, uint32 sequence_size, uint32 prefill_size = 0u, sdouble32 expected_label = double_literal(0.0));

} /* namespace rafko_test */

#endif /* TEST_UTILITY_H */
