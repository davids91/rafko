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

#include "rafko_net/services/partial_solution_builder.hpp"

#include <stdexcept>

#include "rafko_utilities/services/rafko_math_utils.hpp"
#include "rafko_net/models/input_function.hpp"

namespace rafko_net{

std::pair<std::uint32_t,std::uint32_t> PartialSolutionBuilder::add_neuron_to_partial_solution(const RafkoNet& net, std::uint32_t neuron_index){
  RFASSERT_LOG("Adding Neuron[{}] to partial solution!", neuron_index);
  m_previousNeuronInputIndex = m_inputSynapse.cached_size();
  m_previousNeuronInputSource = m_neuronInputNone;
  m_partialInputSynapseCount = 0;
  m_neuronSynapseCount = 0;

  if(net.neuron_array_size() > static_cast<int>(neuron_index)){
    std::uint32_t max_reach_back = 0;
    std::uint32_t max_reach_index = 0;
    SynapseIterator<> weight_iterator(net.neuron_array(neuron_index).input_weights());
    SynapseIterator<InputSynapseInterval> input_iterator(net.neuron_array(neuron_index).input_indices());
    m_partial.mutable_output_data()->set_interval_size(m_partial.output_data().interval_size() + 1u);

    RFASSERT_LOG("Network Input synapses number for Neuron: {}", net.neuron_array(neuron_index).input_indices_size());
    RFASSERT_LOG("Network Weight synapses number for Neuron: {}", net.neuron_array(neuron_index).input_weights_size());

    /* Copy in Neuron parameters and weights from the net */
    m_partial.add_neuron_input_functions(net.neuron_array(neuron_index).input_function());
    m_partial.add_neuron_transfer_functions(net.neuron_array(neuron_index).transfer_function());
    m_partial.add_neuron_spike_functions(net.neuron_array(neuron_index).spike_function());
    m_partial.add_weight_synapse_number(net.neuron_array(neuron_index).input_weights_size());
    weight_iterator.iterate([&](IndexSynapseInterval weight_synapse){
      m_partial.add_weight_indices()->set_starts(m_partial.weight_table_size());
      m_partial.mutable_weight_indices(m_partial.weight_indices_size() - 1u)->set_interval_size(weight_synapse.interval_size());
    },[&](std::int32_t weight_index){
      m_partial.add_weight_table(net.weight_table(weight_index));
    });

    /* Copy in input data references */
    m_neuronSynapseCount = 0;
    m_previousNeuronInputSource = m_neuronInputNone;
    m_previousNeuronInputIndex = m_inputSynapse.cached_size(); /* set input value to point above the size of the input */
    const std::uint32_t index_synapse_previous_size = m_partial.inside_indices_size();

    std::uint32_t current_backreach;
    input_iterator.iterate([&](InputSynapseInterval interval_synapse){
      RFASSERT_LOG("Input synapse reach past loops: {}", interval_synapse.reach_past_loops());
      current_backreach = interval_synapse.reach_past_loops();
      if(interval_synapse.reach_past_loops() > max_reach_back)
        max_reach_back = interval_synapse.reach_past_loops();
      if(SynapseIterator<InputSynapseInterval>::is_synapse_input(interval_synapse)){
        std::uint32_t input_index = SynapseIterator<InputSynapseInterval>::external_index_from_array_index(interval_synapse.starts()) + interval_synapse.interval_size() - 1u;
        if(max_reach_index < input_index)max_reach_index = input_index;
      }
    },[&](std::int32_t neuron_input_index){ /* Put each Neuron input into the @PartialSolution */
      if(!look_for_neuron_input(neuron_input_index, current_backreach)){ /* Neuron input was found in the @PartialSolution inputs, continue to look for it.. */
        /* Check if any synapses needs to be closed */
        if( /* if the Neuron has current inputs from the past or inputs are not found internally */
          (0 < current_backreach)||(!look_for_neuron_input_internally(neuron_input_index))
        ){
          if( /* Close input synapse if */
            (0 < m_partialInputSynapseCount) /* There is one open already */
            &&(( /* The latest index in the input synapse isn't the preceeding index of the current index */
                SynapseIterator<>::is_index_input(neuron_input_index)
                &&(m_inputSynapse.back() != (neuron_input_index + 1))
              )||(
                (!SynapseIterator<>::is_index_input(neuron_input_index))
                &&(m_inputSynapse.back() != (neuron_input_index - 1))
              )||(m_inputSynapse.last_synapse().reach_past_loops() != current_backreach) /* Current index not in same memory depth */
            )
          ){
            m_partialInputSynapseCount = 0; /* Close synapse! */
          }
          if(0 < m_neuronSynapseCount){
            if(
              (m_neuronInputExternal != m_previousNeuronInputSource)
              ||(static_cast<int>(m_inputSynapse.cached_size() - 1) != m_previousNeuronInputIndex)
            )m_neuronSynapseCount = 0; /* Close synapse! */
          }
          m_previousNeuronInputIndex = m_inputSynapse.cached_size(); /* Update previous neuron input source as well */
          m_previousNeuronInputSource = m_neuronInputExternal;/* since the input was added to be taken from the @PartialSolution inputs */
          add_to_synapse( /* Neural input shall be added from the input of the @PartialSolution */
            SynapseIterator<>::external_index_from_array_index(m_inputSynapse.cached_size()), 0,
            m_neuronSynapseCount, m_partial.mutable_inside_indices()
          );
          add_to_synapse(neuron_input_index, current_backreach, m_partialInputSynapseCount, m_partial.mutable_input_data());
          m_inputSynapse.refresh_cached_size();
          RFASSERT_LOG("Extending partial input with: [{}:{}]", neuron_input_index, current_backreach);
        }/* Neuron input was found internally in the @PartialSolution */
      }/*if(Neuron input was not found in the partial inputs)*/
    });

    RFASSERT_LOG("Partial solution Input synapses number for Neuron: {}", (m_partial.inside_indices_size() - index_synapse_previous_size));
    RFASSERT_LOG("partial.inside_indices_size(): {}", m_partial.inside_indices_size());

    if(0 < (m_partial.inside_indices_size() - index_synapse_previous_size))
      m_partial.add_index_synapse_number(m_partial.inside_indices_size() - index_synapse_previous_size);

    if( /* In case th latest input synapse is of 0 length, remove it */
      (0 < m_partial.input_data_size())
      &&(0 == m_partial.input_data(m_partial.input_data_size()-1).interval_size())
    ){
      m_partial.mutable_input_data()->RemoveLast();
      /*!Note: Since the last synapse was empty, size will not change by removing it, so the below line need not to be called. */
      /* input_synapse.refresh_cached_size(); */
    }

    return std::make_pair(max_reach_back, max_reach_index);
  }else throw std::runtime_error("Neuron index is out of bounds from net neuron array!");
}

bool PartialSolutionBuilder::look_for_neuron_input(std::int32_t neuron_input_index, std::uint32_t input_reach_back){
  if( /* In case there are already inputs present.. */
    (1u < m_inputSynapse.cached_size()) /* ..and the previously found input is among them */
    &&(m_previousNeuronInputSource == m_neuronInputExternal)
    &&(m_previousNeuronInputIndex < static_cast<std::int32_t>(m_inputSynapse.cached_size() - 1u))
    &&(neuron_input_index == m_inputSynapse[m_previousNeuronInputIndex + 1u])
    &&(input_reach_back == m_inputSynapse.reach_past_loops<InputSynapseInterval>(m_previousNeuronInputIndex + 1u))
  ){/* ..and the input index currently in search is the next one in the input synapse */
    ++m_previousNeuronInputIndex;
    /* previous_neuron_input_source = neuron_input_external; implicitly implied.. */
    add_to_synapse(
      SynapseIterator<>::external_index_from_array_index(m_previousNeuronInputIndex),
      0, m_neuronSynapseCount, m_partial.mutable_inside_indices()
    );
    return true;
  }
  std::uint32_t candidate_index_inside_input = m_inputSynapse.cached_size();
  auto cache_hit = m_foundNetworkInputInPartialInput.find(
    rafko_utilities::pair_hash({neuron_input_index,input_reach_back})
  );
  if(cache_hit == m_foundNetworkInputInPartialInput.end()){
    std::uint32_t current_backreach;
    m_inputSynapse.iterate_terminatable([&current_backreach](InputSynapseInterval interval_synapse){
      current_backreach = interval_synapse.reach_past_loops();
      return true;
    },[&](std::int32_t synapse_index){
      if(candidate_index_inside_input == m_inputSynapse.cached_size()) candidate_index_inside_input = 0u;
      if((input_reach_back == current_backreach)&&(synapse_index == neuron_input_index)){ /* If the index as well as the time of input matches */
        cache_hit = std::get<0>(m_foundNetworkInputInPartialInput.insert({
          rafko_utilities::pair_hash({neuron_input_index,input_reach_back}), candidate_index_inside_input
        }));
        return false; /* No need to continue Synapse iteration, found the right candidate! */
      }else{ /* Step the candidate iterator forward to the next index in the input array */
        ++candidate_index_inside_input;
        return true;
      }
    });
  }else candidate_index_inside_input = cache_hit->second; /* found in cache */
  if(cache_hit != m_foundNetworkInputInPartialInput.end()){ /* Found the neuron input in the candidate synapse inputs */
    RFASSERT_LOG("Input synapse cached size: {} vs size: {}", m_inputSynapse.cached_size(), m_inputSynapse.size());
    RFASSERT(candidate_index_inside_input < m_inputSynapse.cached_size());
    if(0 < m_neuronSynapseCount){
      if( /* Check if the newly added Neuron synapse can be continued based on value, or a new Synapse needs to be added */
        (m_neuronInputExternal != m_previousNeuronInputSource)
        ||(static_cast<int>(candidate_index_inside_input-1) != m_previousNeuronInputIndex)
      )m_neuronSynapseCount = 0; /* Close synapse! */
    }
    m_previousNeuronInputIndex = candidate_index_inside_input;
    m_previousNeuronInputSource = m_neuronInputExternal;
    add_to_synapse( /* inside indices always taking input from the current value */
      SynapseIterator<>::external_index_from_array_index(candidate_index_inside_input),
      0, m_neuronSynapseCount, m_partial.mutable_inside_indices()
    );
    return true;
  }else return false; /* couldn't find the Neuron input in the @Partial solution input synapses */
}

bool PartialSolutionBuilder::look_for_neuron_input_internally(std::uint32_t neuron_input_index){
  if(
    (static_cast<std::int32_t>(neuron_input_index) >= m_partial.output_data().starts())
    &&(neuron_input_index < (m_partial.output_data().starts() + m_partial.output_data().interval_size()))
  ){
    const std::uint32_t inner_neuron_index = (neuron_input_index - m_partial.output_data().starts());
    if( /* there is a synapse already open for the current Neuron input */
      (0 < m_neuronSynapseCount)
      &&( /* ..and the current found index can not continue it */
        (m_neuronInputInternal != m_previousNeuronInputSource)
        ||(static_cast<int>(inner_neuron_index)-1 != m_previousNeuronInputIndex)
      )
    )m_neuronSynapseCount = 0; /* Close synapse! */
    m_previousNeuronInputIndex = inner_neuron_index;
    m_previousNeuronInputSource = m_neuronInputInternal;
    add_to_synapse( /* The Neuron input points to an internal Neuron (no conversion to input synapse index) */
      inner_neuron_index, 0, m_neuronSynapseCount, m_partial.mutable_inside_indices()
    );
    return true;
  }else return false;
}

void PartialSolutionBuilder::add_to_synapse(std::int32_t index, std::uint32_t reach_back, std::uint32_t& current_synapse_count, google::protobuf::RepeatedPtrField<InputSynapseInterval>* synapse_intervals){
  if((0 < synapse_intervals->size())&&(0 < current_synapse_count)){ /* Currently building a synapse already */
    ++current_synapse_count;
    synapse_intervals->Mutable(synapse_intervals->size()-1)->set_interval_size(current_synapse_count);
  }else{ /* Opening up a totally new Neuron Synapse */
    InputSynapseInterval& new_interval = *synapse_intervals->Add();
    new_interval.set_starts(index);
    new_interval.set_interval_size(1);
    new_interval.set_reach_past_loops(reach_back);
    current_synapse_count = 1;
  }
}

} /* namespace rafko_net */
