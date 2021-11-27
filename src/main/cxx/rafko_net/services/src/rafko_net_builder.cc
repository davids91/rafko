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

#include "rafko_net/services/rafko_net_builder.h"

#include <time.h>

#include <stdexcept>

#include "rafko_net/models/dense_net_weight_initializer.h"

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net {

using std::shared_ptr;

RafkoNet* RafkoNetBuilder::dense_layers(vector<uint32> layer_sizes){

  using std::make_shared;

  uint32 previous_size = 0;
  uint32 numNeurons = 0;
  InputSynapseInterval temp_input_interval;
  IndexSynapseInterval temp_index_interval;
  /* Calculate number of weights needed overall
   * - Input Layer shall have a weight for every input for every neuron
   * - Input Layer shall have a weight for every bias and memory_filter for every neuron
   */
  uint64 numWeights = (layer_sizes[0] * arg_input_size) + (2 * layer_sizes[0]); /* The first layer only takes input from the @RafkoNet input data */
  for(uint32 layerSize : layer_sizes){
    if(0 != numNeurons){ /* The first layer is already included in numWeights */
      numWeights += previous_size * layerSize; /* Calculate the number of weights needed */
      numWeights += layerSize * 2; /* Every neuron shall store its bias and memory_filter amongst the weights */
    }
    if(network_recurrence_to_self == recurrence)numWeights += layerSize; /* Recurrence to self */
    else if(network_recurrence_to_layer == recurrence)numWeights += (layerSize * layerSize); /* Recurrence to layer */
    previous_size = layerSize;
    numNeurons += layerSize; /* Calculate the number of elements needed */
  }

  if( /* Required arguments are set */
    (is_input_size_set && is_expected_input_range_set)
    &&(
      (!is_output_neuron_number_set) /* Output size is either not set */
      ||(arg_output_neuron_number == layer_sizes.back()) /* Or compliant to the Dense layer */
    )
  ){
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(context.get_arena_ptr());
    uint32 layer_input_starts_at = 0;
    uint64 weightIt = 0;
    uint64 neurIt = 0;
    sdouble32 expPrevLayerOutput = TransferFunction::get_average_output_range(transfer_function_identity);

    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(std::make_shared<DenseNetWeightInitializer>(context));
    }

    arg_weight_table = vector<sdouble32>(numWeights);
    arg_neuron_array = vector<Neuron>(numNeurons);

    previous_size = arg_input_size;
    for(uint32 layerIt = 0; layerIt < layer_sizes.size(); layerIt++)
    { /* Create the Dense Layers */

      /* Configuring the weight_initializerializer for this layer */
      arg_weight_initer->set(
        (0 == layerIt)?(arg_input_size):(layer_sizes[layerIt-1]),
        expPrevLayerOutput
      );

      /* Add the Neurons */
      expPrevLayerOutput = 0;
      for(uint32 layerNeurIt = 0; layerNeurIt < layer_sizes[layerIt]; layerNeurIt++){
        if(is_allowed_transfer_functions_by_layer_set){
          arg_neuron_array[neurIt].set_transfer_function_idx(
            TransferFunction::next(arg_allowed_transfer_functions_by_layer[layerIt])
          );
        }else{
          arg_neuron_array[neurIt].set_transfer_function_idx(TransferFunction::next());
        }

        /* Storing the expected output of this Net */
        if(0 < layerIt)expPrevLayerOutput += TransferFunction::get_average_output_range(
          arg_neuron_array[neurIt].transfer_function_idx()
        );

        /* Add the previous layer to the built net */
        temp_index_interval.set_starts(weightIt);
        temp_index_interval.set_interval_size(previous_size + 1u); /* Previous layer + the spike function weight */
        *arg_neuron_array[neurIt].add_input_weights() = temp_index_interval;

        /* Add the spike function weight as the first weight of the Neuron */
        arg_weight_table[weightIt] = arg_weight_initer->next_memory_filter();
        ++weightIt;

        if(0 == layerIt){
          temp_input_interval.set_starts(SynapseIterator<>::synapse_index_from_input_index(0));
        }else{
          temp_input_interval.set_starts(layer_input_starts_at);
        }
        temp_input_interval.set_interval_size(previous_size);
        temp_input_interval.set_reach_past_loops(0);
        *arg_neuron_array[neurIt].add_input_indices() = temp_input_interval;

        /* Add the input weights for the previous layer */
        for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < previous_size; neuron_weight_iterator++){
          arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
            arg_neuron_array[neurIt].transfer_function_idx()
          );
          weightIt++;
        }

        /* Add recurrence of the Neuron */
        if(network_recurrence_to_self == recurrence){ /* recurrence to self */
          /* Add the weight synapse */
          temp_index_interval.set_starts(weightIt);
          temp_index_interval.set_interval_size(1 + 1); /* self-recurrence + a bias */
          *arg_neuron_array[neurIt].add_input_weights() = temp_index_interval;

          /* Add the weight */
          arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
            arg_neuron_array[neurIt].transfer_function_idx()
          );
          weightIt++;

          /* Add the input synapse */
          temp_input_interval.set_starts(neurIt); /* self-recurrence, an additional input snypse */
          temp_input_interval.set_interval_size(1); /* of a lone input as the actual @Neuron itself */
          temp_input_interval.set_reach_past_loops(1);
          *arg_neuron_array[neurIt].add_input_indices() = temp_input_interval;
        }else if(network_recurrence_to_layer == recurrence){ /* recurrence to layer */
          /* Add the weight synapse */
          temp_index_interval.set_starts(weightIt);
          temp_index_interval.set_interval_size(layer_sizes[layerIt] + 1); /* Current layer + a bias */
          *arg_neuron_array[neurIt].add_input_weights() = temp_index_interval;

          /* Add the weights */
          for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < layer_sizes[layerIt]; neuron_weight_iterator++){
            arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
              arg_neuron_array[neurIt].transfer_function_idx()
            );
            weightIt++;
          }

          /* Add the input synapse */
          if(0 < layerIt)temp_input_interval.set_starts(layer_input_starts_at + layer_sizes[layerIt-1]); /* starts at the beginning of the current layer */
          else temp_input_interval.set_starts(layer_input_starts_at); /* starts at the beginning of the current layer */

          temp_input_interval.set_interval_size(layer_sizes[layerIt]); /* takes up the whole layer */
          temp_input_interval.set_reach_past_loops(1);
          *arg_neuron_array[neurIt].add_input_indices() = temp_input_interval;
        }else{ /* Only bias */
          temp_index_interval.set_starts(weightIt);
          temp_index_interval.set_interval_size(1); /* a lone bias weight */
          *arg_neuron_array[neurIt].add_input_weights() = temp_index_interval;
        }

        arg_weight_table[weightIt] = arg_weight_initer->next_bias();
        weightIt++;
        neurIt++; /* Step the neuron iterator forward */
      }

      if(0 == layerIt){
        expPrevLayerOutput = arg_expected_input_range;
        layer_input_starts_at = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layer_sizes[layerIt]);
        layer_input_starts_at += previous_size;
      }
      previous_size = layer_sizes[layerIt];
    } /* Itearte through all of the layers */

    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw std::runtime_error("Input Output Pre-requisites failed;Unable to determine Net Structure!");
}

RafkoNet* RafkoNetBuilder::build(){
  if( /* Required arguments are set */
    (is_input_size_set && is_output_neuron_number_set)
    &&(is_neuron_array_set && is_weight_table_set) /* needed arguments are set */
    &&(0 < arg_weight_table.size())&&(0 < arg_neuron_array.size()) /* There are at least some Neurons and Weights */
    &&(arg_output_neuron_number <= arg_neuron_array.size()) /* Output size isn't too big */
  ){
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(context.get_arena_ptr());
    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(arg_output_neuron_number);
    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw std::runtime_error("Inconsistent parameters given to Sparse Net Builder!");
}

} /* namespace rafko_net */
