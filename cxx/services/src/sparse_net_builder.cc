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

#include "services/sparse_net_builder.h"

#include "models/dense_net_weight_initializer.h"
#include "services/synapse_iterator.h"

#include <stdexcept>
#include <time.h>

namespace sparse_net_library {

using std::shared_ptr;

SparseNet* Sparse_net_builder::dense_layers(vector<uint32> layer_sizes){

  using std::make_shared;

  uint32 previous_size = 0;
  uint32 numNeurons = 0;
  Input_synapse_interval temp_input_interval;
  Index_synapse_interval temp_index_interval;
  /* Calculate number of weights needed overall
   * - Input Layer shall have a weight for every input for every neuron
   * - Input Layer shall have a weight for every bias and memory_filter for every neuron
   */
  uint64 numWeights = (layer_sizes[0] * arg_input_size) + (2 * layer_sizes[0]); /* The first layer only takes input from the @SparseNet input data */
  for(uint32 layerSize : layer_sizes){
    if(0 != numNeurons){ /* The first layer is already included in numWeights */
      numWeights += previous_size * layerSize; /* Calculate the number of weights needed */
      numWeights += layerSize * 2; /* Every neuron shall store its bias and memory_filter amongst the weights */
    }
    if(0x01 == recurrence)numWeights += layerSize; /* Recurrence to self */
    else if(0x02 == recurrence)numWeights += (layerSize * layerSize); /* Recurrence to layer */
    previous_size = layerSize;
    numNeurons += layerSize; /* Calculate the number of elements needed */
  }

  if( /* Required arguments are set */
    (is_input_size_set && is_expected_input_range_set && is_cost_function_set)
    &&(
      (!is_output_neuron_number_set) /* Output size is either not set */
      ||(arg_output_neuron_number == layer_sizes.back()) /* Or compliant to the Dense layer */
    )
  ){
    SparseNet* ret(google::protobuf::Arena::CreateMessage<SparseNet>(arg_arena));
    uint32 layerStart = 0;
    uint64 weightIt = 0;
    uint64 neurIt = 0;
    sdouble32 expPrevLayerOutput = Transfer_function::get_average_output_range(TRANSFER_FUNCTION_IDENTITY);

    ret->set_cost_function(arg_cost_function);
    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(std::make_shared<Dense_net_weight_initializer>());
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
        arg_weight_table[weightIt] = arg_weight_initer->next_memory_filter();
        arg_neuron_array[neurIt].set_memory_filter_idx(weightIt);
        ++weightIt;
        if(is_allowed_transfer_functions_by_layer_set){
          arg_neuron_array[neurIt].set_transfer_function_idx(
            Transfer_function::next(arg_allowed_transfer_functions_by_layer[layerIt])
          );
        }else{
          arg_neuron_array[neurIt].set_transfer_function_idx(Transfer_function::next());
        }

        /* Storing the expected output of this Net */
        if(0 < layerIt)expPrevLayerOutput += Transfer_function::get_average_output_range(
          arg_neuron_array[neurIt].transfer_function_idx()
        );

        /* Add the previous layer to the built net */
        temp_index_interval.set_starts(weightIt);
        temp_index_interval.set_interval_size(previous_size + 1); /* Previous layer + a bias */
        *arg_neuron_array[neurIt].add_input_weights() = temp_index_interval;

        if(0 == layerIt){
          temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0));
        }else{
          temp_input_interval.set_starts(layerStart);
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
        if(0x01 == recurrence){ /* recurrence to self */
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
        }else if(0x02 == recurrence){ /* recurrence to layer */
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
          temp_input_interval.set_starts(layerStart); /* starts at the beginning of the current layer */
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
        layerStart = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layer_sizes[layerIt]);
        layerStart += previous_size;
      }
      previous_size = layer_sizes[layerIt];
    } /* Itearte through all of the layers */

    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw std::runtime_error("Input Output Pre-requisites failed;Unable to determine Net Structure!");
}

SparseNet* Sparse_net_builder::build(void){
  if( /* Required arguments are set */
    (is_input_size_set && is_output_neuron_number_set && is_cost_function_set)
    &&(is_neuron_array_set && is_weight_table_set) /* needed arguments are set */
    &&(0 < arg_weight_table.size())&&(0 < arg_neuron_array.size()) /* There are at least some Neurons and Weights */
    &&(arg_output_neuron_number <= arg_neuron_array.size()) /* Output size isn't too big */
  ){
    SparseNet* ret = google::protobuf::Arena::CreateMessage<SparseNet>(arg_arena);
    ret->set_cost_function(arg_cost_function);
    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(arg_output_neuron_number);
    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw std::runtime_error("Inconsistent parameters given to Sparse Net Builder!");
}

} /* namespace sparse_net_library */
