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

#include <memory>
#include <vector>

#include "rafko_global.hpp"

#include "rafko_gym/models/rafko_dataset_implementation.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_protocol/training.pb.h"

namespace rafko_test {

/**
 * @brief     Cross-platform method of obtaining the width of the actual console
 */
extern std::uint32_t get_console_width();

/**
 * @brief      generates a partial partial_solution manually based on the Neural
 * Network structure: 2 Neurons: The first neuron has the inputs and the second
 * has the first neuron
 *
 * @param      partial_solution  The partial solution
 * @param[in]  number_of_inputs  The number of inputs to the network
 */
extern void
manual_2_neuron_partial_solution(rafko_net::PartialSolution &partial_solution,
                                 std::uint32_t number_of_inputs,
                                 std::uint32_t neuron_offset = 0);

/** @brief Calculates the result of the partial partial_solution manually based
 * on the structure provided by @manual_2_neuron_partial_solution In case there
 * are more than 2 inputs, all of them shall be processed with the same weight
 *         The end result shall be calculated as follows:
 *         - result1 = f[0]((input1 * weight1 + input2 * weight2) + bias1)
 * //weighted inputs + bias, given to the transfer function
 *         - result1 = (prev_result1 * memory_filter1) + (result1 *
 * ((1.0)-memory_rato1) //apply memory filter
 *         - result2 = f[1]((result1 * weight3) + bias2)
 *         - end_result = (prev_result2 * memory_filter2) + (result2 *
 * ((1.0)-memory_rato2)
 * @param[in]  partial_inputs      The collected inputs of the @PartialSolution
 * @param      prev_neuron_output  The previous neuron data
 * @param[in]  partial_solution    The partial solution containing the weights
 * andmisc parameters of the calculation
 * @param[in]  neuron_offset       THe neuron offset in which the partial
 * solution is present in the whole network
 */
extern void
manual_2_neuron_result(const std::vector<double> &partial_inputs,
                       std::vector<double> &prev_neuron_output,
                       const rafko_net::PartialSolution &partial_solution,
                       std::uint32_t neuron_offset = 0);

/**
 * @brief      Calculates the result of a fully connected Network for the given
 * inputs
 *
 * @param[in]  inputs           The inputs
 * @param      previous_data    The data the Network should recieve from the
 * previous run
 * @param[in]  neuron_data      The data the Neural network should contain in
 * the current run
 * @param[in]  layer_structure  The layer structure
 * @param[in]  network          The network
 */
extern void manaual_fully_connected_network_result(
    std::vector<double> &inputs, std::vector<double> previous_data,
    std::vector<double> &neuron_data,
    std::vector<std::uint32_t> layer_structure, rafko_net::RafkoNet network);

/**
 * @brief      Checks if the inputs are pointing to the same data and weight
 * values are matching in the given
 *             @RafkoNet and @Solution
 *
 * @param[in]  net       The net
 * @param      solution  The solution
 */
extern void check_if_the_same(const rafko_net::RafkoNet &net,
                              const rafko_net::Solution &solution);

/**
 * @brief      Prints weights for the given arguments
 *
 * @param[in]  net       The net
 * @param      solution  The solution
 */
extern void print_weights(const rafko_net::RafkoNet &net,
                          const rafko_net::Solution &solution);

/**
 * @brief      Prints a training sample of the given data set, under the given
 * index. Expects 2 inputs and one output!
 *
 * @param[in]  sample_sequence_index  The sample sequence index
 * @param      data_set               The data set
 * @param[in]  net                    The net
 * @param[in]  settings               The service settings
 */
extern void
print_training_sample(std::uint32_t sample_sequence_index,
                      rafko_gym::RafkoDatasetImplementation &data_set,
                      const rafko_net::RafkoNet &net,
                      const rafko_mainframe::RafkoSettings &settings);

/**
 * @brief      Creates a normalized dataset for addition: basically adding two
 * numbers together. The generated dataset is adequate for testing non-recurrent
 * neural networks
 *
 * @param[in]  number_of_samples  The number of samples to create
 *
 * @return     The addition dataset. For each sample: Inputs: [a][b]; Outputs:
 * [a+b]
 */
extern std::pair<std::vector<std::vector<double>>,
                 std::vector<std::vector<double>>>
create_addition_dataset(std::uint32_t number_of_samples);

/**
 * @brief      Creates a normalized dataset for adding binary numbers: each
 * number is stored as sequences of 0/1 one after another. The generated dataset
 * is adequate for testing recurrent neural networks
 *
 * @param[in]  number_of_samples  The number of samples to create
 * @param[in]  sequence_size      The number of sequences one sample should
 * contain, i.e. the size of the binary number
 *
 * @return     The addition dataset. For each sample: Inputs:
 * [[a0][...][an]][[b0][...][bn]]; Labels: [[result0][...][resultn]]
 */
extern std::pair<std::vector<std::vector<double>>,
                 std::vector<std::vector<double>>>
create_sequenced_addition_dataset(std::uint32_t number_of_samples,
                                  std::uint32_t sequence_size);

/**
 * @brief      Checks if the two arguments match or not
 *
 * @param      sample_data      The sample data
 * @param      ringbuffer_data  The ringbuffer data
 */
extern void check_data_match(std::vector<double> &sample_data,
                             std::vector<double> &ringbuffer_data);

/**
 * @brief      Generates a random Fully connected Dense network with some Layers
 * set to have the softmax feature
 *
 * @param[in]  input_size       The size of the input vector accepted by the
 * produces network
 * @param      settings         Contextual information
 * @param[in]  output_size      The size of the networks expected output. It is
 * decided randomly, when the given value is 0.
 *
 * @return     the generated network to be owned by the caller
 */
extern rafko_net::RafkoNet *generate_random_net_with_softmax_features(
    std::uint32_t input_size, rafko_mainframe::RafkoSettings &settings,
    std::uint32_t output_size = 0);

/**
 * @brief      Generates a random Fully connected Dense network with some Layers
 * set to have the softmax feature; And some Neurons + Layers having inputs from
 * the past
 *
 * @param[in]  input_size       The size of the input vector accepted by the
 * produces network
 * @param[in]  settings         Contextual information
 * @param[in]  output_size      The size of the networks expected output. It is
 * decided randomly, when the given value is 0.
 *
 * @return     the generated network to be owned by the caller
 */
extern rafko_net::RafkoNet *
generate_random_net_with_softmax_features_and_recurrence(
    std::uint32_t input_size, rafko_mainframe::RafkoSettings &settings,
    std::uint32_t output_size = 0);

/**
 * @brief      Creates a random dataset based on the given parameters
 *
 * @param[in]  input_size         The number of expected inputs are stored for
 * the @DataSetPackage
 * @param[in]  feature_size       The number of features and labels to be stored
 * inside the @DataSetPackage
 * @param[in]  sample_number      The number of samples to generate
 * @param[in]  sequence_size      The number of sequences one sample should
 * contain
 * @param[in]  expected_label     The content of the dataset repeated in the
 * correct structure
 *
 * @return     The created dataset
 */
extern std::unique_ptr<rafko_gym::DataSetPackage>
create_dataset(std::uint32_t input_size, std::uint32_t feature_size,
               std::uint32_t sample_number, std::uint32_t sequence_size,
               std::uint32_t prefill_size = 0u, double expected_label = (0.0),
               double label_delta_per_feature = (0.0));

} /* namespace rafko_test */

#endif /* TEST_UTILITY_H */
