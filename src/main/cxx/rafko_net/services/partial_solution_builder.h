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

#include "rafko_global.h"

#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net{

/**
 * @brief      Front-end to create partial solution objects by adding Neurons into them.
 *             Weights of a Neuron consists of: {memory_ratio, w1..wn, bias1..biasn}
 */
class RAFKO_FULL_EXPORT PartialSolutionBuilder{
public:
  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution input,
   *             and adds the input to it if found
   *
   * @param[in]  net            The network to read the Neuron from
   * @param[in]  neuron_index   The index of the Neuron to read inside the @RafkoNet
   * @param      partial        The reference of the partial solution to add the Neuron into
   *
   * @return     returns the input parameters of the Neurons: {maximum reach back in Neural memory, maximum input index reached}
   */
  static std::pair<uint32,uint32> add_neuron_to_partial_solution(const RafkoNet& net, uint32 neuron_index, PartialSolution& partial);

private:
  /**
   * @brief      Adds the given index to the given synapse
   *
   * @param[in]  index                  The Neuron index
   * @param[in]  reach_back             The input reach_back value to show how far the input reaches back to past runs
   * @param      current_synapse_count  The number of elements currently present in the synapse
   * @param      synapse_intervals      The array of synapses to add the index to
   */
  static void add_to_synapse(sint32 index, uint32 reach_back, uint32& current_synapse_count, google::protobuf::RepeatedPtrField<InputSynapseInterval>* synapse_intervals);

  /**
   * @brief      Adds the given index to the given synapse
   *
   * @param[in]  index                  The Neuron or Weight index
   * @param      current_synapse_count  The number of elements currently present in the synapse
   * @param      synapse_intervals      The array of synapses to add the index to
   */
  static void add_to_synapse(sint32 index, uint32& current_synapse_count, google::protobuf::RepeatedPtrField<IndexSynapseInterval>* synapse_intervals);

  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution input,
   *             and adds the input to it if found
   *
   * @param[in]  neuron_input_index  The neuron input index to look for
   * @param[in]  input_reach_back    The number of loops a Neural input reaches back to
   *
   * @return     returns true if the neuron index was found in the @PartialSolution input
   */
  static bool look_for_neuron_input(
    sint32 neuron_input_index, uint32 input_reach_back,
    SynapseIterator<InputSynapseInterval>& input_synapse, PartialSolution& partial
  );

  /**
   * @brief      Looks for the given Neuron index in the @PartialSolution internally,
   *             and adds the input to it if found
   *
   * @param[in]  neuron_input_index  The neuron input index to look for
   *
   * @return     returns true if the neuron index was found in the @PartialSolution Inner Neurons
   */
  static bool look_for_neuron_input_internally(uint32 neuron_input_index, PartialSolution& partial);

  /**
   * Temporary helper variables used only during Neuron mapping which is started by @add_neuron_to_partial_solution
   * but used additionally in @look_for_neuron_input_internally and @look_for_neuron_input
   */
  static uint32 neuron_synapse_count;
  static uint32 partial_input_synapse_count;
  static sint32 previous_neuron_input_index;
  static uint8 previous_neuron_input_source;

  /**
   *  definitions to assign a source of a neurons input upon building up the partial solution
   */
  static const uint8 neuron_input_none = 0u;
  static const uint8 neuron_input_internal = 1u;
  static const uint8 neuron_input_external = 2u;
};

} /* namespace rafko_net */

#endif /* PARTIAL_SOLUTION_BUILDER_H */
