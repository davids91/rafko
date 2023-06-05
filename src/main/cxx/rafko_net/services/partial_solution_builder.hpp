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

#ifndef PARTIAL_SOLUTION_BUILDER_H
#define PARTIAL_SOLUTION_BUILDER_H

#include "rafko_global.hpp"

#include <unordered_map>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"

#include "rafko_net/services/synapse_iterator.hpp"

namespace rafko_net {

/**
 * @brief      Front-end to create partial solution objects by adding Neurons
 * into them. Weights of a Neuron consists of: {memory_ratio, w1..wn,
 * bias1..biasn}
 */
class RAFKO_EXPORT PartialSolutionBuilder {
public:
  PartialSolutionBuilder(PartialSolution &partial)
      : m_partial(partial), m_inputSynapse(m_partial.input_data()) {}

  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution input,
   *             and adds the input to it if found
   *
   * @param[in]  net            The network to read the Neuron from
   * @param[in]  neuron_index   The index of the Neuron to read inside the
   * @RafkoNet
   *
   * @return     returns the input parameters of the Neurons: {maximum reach
   * back in Neural memory, maximum input index reached}
   */
  [[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
  add_neuron_to_partial_solution(const RafkoNet &net,
                                 std::uint32_t neuron_index);

private:
  PartialSolution &m_partial;
  SynapseIterator<InputSynapseInterval> m_inputSynapse;
  std::unordered_map<std::uint64_t, std::uint32_t>
      m_foundNetworkInputInPartialInput;

  std::uint32_t m_neuronSynapseCount = 0u;
  std::uint32_t m_partialInputSynapseCount = 0u;
  std::int32_t m_previousNeuronInputIndex;
  std::uint8_t m_previousNeuronInputSource;

  /**
   *  definitions to assign a source of a neurons input upon building up the
   * partial solution
   */
  static const std::uint8_t m_neuronInputNone = 0u;
  static const std::uint8_t m_neuronInputInternal = 1u;
  static const std::uint8_t m_neuronInputExternal = 2u;

  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution input,
   *             and adds the input to it if found
   *
   * @param[in]  neuron_input_index  The neuron input index to look for
   * @param[in]  input_reach_back    The number of loops a Neural input reaches
   * back to
   *
   * @return     returns true if the neuron index was found in the
   * @PartialSolution input
   */
  bool look_for_neuron_input(std::int32_t neuron_input_index,
                             std::uint32_t input_reach_back);

  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution
   * internally, and adds the input to it if found
   *
   * @param[in]  neuron_input_index  The neuron input index to look for
   *
   * @return     returns true if the neuron index was found in the
   * @PartialSolution Inner Neurons
   */
  bool look_for_neuron_input_internally(std::uint32_t neuron_input_index);

  /**
   * @brief      Adds the given index to the given synapse
   *
   * @param[in]  index                  The Neuron index
   * @param[in]  reach_back             The input reach_back value to show how
   * far the input reaches back to past runs
   * @param      current_synapse_count  The number of elements currently present
   * in the synapse
   * @param      synapse_intervals      The array of synapses to add the index
   * to
   */
  static void
  add_to_synapse(std::int32_t index, std::uint32_t reach_back,
                 std::uint32_t &current_synapse_count,
                 google::protobuf::RepeatedPtrField<InputSynapseInterval>
                     *synapse_intervals);
};

} /* namespace rafko_net */

#endif /* PARTIAL_SOLUTION_BUILDER_H */
