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

#include <algorithm>
#include <stdexcept>
#include <string>

#include "rafko_net/models/dense_net_weight_initializer.h"
#include "rafko_net/models/input_function.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net {

RafkoNetBuilder& RafkoNetBuilder::set_neuron_input_function(uint32 layer_index, uint32 layer_neuron_index, Input_functions function){
  std::tuple<uint32,uint32,Input_functions> new_element = std::make_tuple(
    layer_index,layer_neuron_index,function
  );
  std::vector<std::tuple<uint32,uint32,Input_functions>>::iterator found_element = std::find_if(
    arg_neuron_index_input_functions.begin(), arg_neuron_index_input_functions.end(),
    [&new_element](const std::tuple<uint32,uint32,rafko_net::Input_functions>& current_element){
      return ( (std::get<0>(new_element) == std::get<0>(current_element))&&(std::get<1>(new_element) == std::get<1>(current_element)) );
    }
  );

  if(found_element == arg_neuron_index_input_functions.end())
    arg_neuron_index_input_functions.push_back(std::move(new_element));
    else  *found_element = std::move(new_element);

  return *this;
}

RafkoNetBuilder& RafkoNetBuilder::add_feature_to_layer(uint32 layer_index, Neuron_group_features feature){
  std::vector<std::pair<uint32,Neuron_group_features>>::iterator it = std::find_if(
    layer_features.begin(), layer_features.end(),
    [layer_index, feature](const std::pair<uint32,Neuron_group_features>& element){
      return( /* only add feature if it's not already inside the network */
        (layer_index == std::get<0>(element))
        &&(feature == std::get<1>(element))
      );
    }
  );
  if(it == layer_features.end()){ /* only append feature group if layer is not al */
    layer_features.push_back(std::make_pair(layer_index, feature));
  }
  return *this;
}

RafkoNet* RafkoNetBuilder::dense_layers(std::vector<uint32> layer_sizes){
  uint32 previous_size = 0;
  uint32 numNeurons = 0;
  InputSynapseInterval temp_input_interval;
  IndexSynapseInterval weight_synapse;

  std::sort(arg_neuron_index_input_functions.begin(),arg_neuron_index_input_functions.end(),
  [](const std::tuple<uint32,uint32,Input_functions>& a, std::tuple<uint32,uint32,Input_functions>& b){
    return std::get<0>(a) < std::get<0>(b);
  });

  if(
    (is_allowed_transfer_functions_by_layer_set)
    &&(layer_sizes.size() != arg_allowed_transfer_functions_by_layer.size())
  )throw std::runtime_error(
    "Allowed transfer functions need to be set for each layer explicitly; sizes don't match: "
    + std::string("layers: ") + std::to_string(layer_sizes.size())
    + std::string(" vs ") + std::to_string(arg_allowed_transfer_functions_by_layer.size())
  );

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
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(settings.get_arena_ptr());
    uint32 layer_input_starts_at = 0;
    uint64 weightIt = 0;
    uint64 neurIt = 0;
    sdouble32 expPrevLayerOutput = TransferFunction::get_average_output_range(transfer_function_identity);

    /* sort the requested features by layer */
    std::sort(layer_features.begin(),layer_features.end(),
    [](const std::pair<uint32,Neuron_group_features>& a, const std::pair<uint32,Neuron_group_features>& b){
      return /* a less, than b */(std::get<0>(a) < std::get<0>(b));
    });

    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = arg_input_size;

    if(!is_weight_initializer_set){
      weight_initializer(std::make_shared<DenseNetWeightInitializer>(settings));
    }

    arg_weight_table = std::vector<sdouble32>(numWeights);
    arg_neuron_array = std::vector<Neuron>(numNeurons);

    previous_size = arg_input_size;
    for(uint32 layer_index = 0; layer_index < layer_sizes.size(); layer_index++)
    { /* Create the Dense Layers */
      while( /* erase all the deprecated input function settings for Neurons based on the layer index */
        ( 0u < arg_neuron_index_input_functions.size() )
        &&( layer_index > std::get<0>(arg_neuron_index_input_functions.front()) )
      ){
        arg_neuron_index_input_functions.erase(arg_neuron_index_input_functions.begin());
      }

      /* Configuring the weight_initializerializer for this layer */
      arg_weight_initer->set(
        (0 == layer_index)?(arg_input_size):(layer_sizes[layer_index-1]),
        expPrevLayerOutput
      );

      /* Store the features for this layer */
      while( (0u < layer_features.size()) && (std::get<0>(layer_features.front()) == layer_index) && (0 < layer_sizes[layer_index]) ){
        FeatureGroup feature_to_add;
        feature_to_add.set_feature(std::get<1>(layer_features.front()));
        feature_to_add.add_relevant_neurons()->set_interval_size(layer_sizes[layer_index]);
        feature_to_add.mutable_relevant_neurons(0)->set_starts(neurIt);
        *ret->add_neuron_group_features() = feature_to_add;
        layer_features.erase(layer_features.begin());
      }

      /* Sort the input function requests based on Neuron indices this time */
      std::sort( /* starting from the beginning of the array.. */
        arg_neuron_index_input_functions.begin(), /* ..because builder continually removes the irrelevant front parts.. */
        std::find_if(arg_neuron_index_input_functions.begin(),arg_neuron_index_input_functions.end(),
          [layer_index](const std::tuple<uint32,uint32,Input_functions>& element){
            return std::get<0>(element) == (layer_index + 1u); /* ..until the part of the array which starts at the next layer. */
          }
          /*!Note: this works because the vector is already sorted based on layer index,
           * so the element in this `find_if` is either the end of the vector
           * or the start of the next layer relevant part of it
           */
        ),
        [](const std::tuple<uint32,uint32,Input_functions>& a, std::tuple<uint32,uint32,Input_functions>& b){
          return std::get<1>(a) < std::get<1>(b); /* the relevant(to the layer_index)y part of the vector is sorted based on Neuron index */
        }
      );

      /* Add the Neurons */
      expPrevLayerOutput = 0;
      for(uint32 layer_neuron_index = 0; layer_neuron_index < layer_sizes[layer_index]; layer_neuron_index++){

        if(
          ( 0u < arg_neuron_index_input_functions.size() )
          &&(layer_neuron_index == std::get<1>(arg_neuron_index_input_functions.front()))
        ){ /* if a Neuron has its input function explicitly set */
          arg_neuron_array[neurIt].set_input_function( InputFunction::next(
            { std::get<2>(arg_neuron_index_input_functions.front()) }
          ) );
        }else arg_neuron_array[neurIt].set_input_function(InputFunction::next());

        while( /* erase all the deprecated input function settings for Neurons based on Neuron index inside the layer */
          ( 0u < arg_neuron_index_input_functions.size() )
          &&(layer_index == std::get<0>(arg_neuron_index_input_functions.front()))
          &&(layer_neuron_index >= std::get<1>(arg_neuron_index_input_functions.front()))
        ){
          arg_neuron_index_input_functions.erase(arg_neuron_index_input_functions.begin());
        }

        if(is_allowed_transfer_functions_by_layer_set){
          arg_neuron_array[neurIt].set_transfer_function(
            TransferFunction::next(arg_allowed_transfer_functions_by_layer[layer_index])
          );
        }else{
          arg_neuron_array[neurIt].set_transfer_function(TransferFunction::next());
        }

        /* Storing the expected output of this Net */
        if(0 < layer_index)expPrevLayerOutput += TransferFunction::get_average_output_range(
          arg_neuron_array[neurIt].transfer_function()
        );

        /* Add the weights for the previous layer to the built net */
        weight_synapse.set_starts(weightIt);
        weight_synapse.set_interval_size(previous_size + 1u); /* Previous layer + the spike function weight */

        /* Add the spike function weight as the first weight of the Neuron */
        arg_weight_table[weightIt] = arg_weight_initer->next_memory_filter();
        ++weightIt;

        if(0 == layer_index){
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
            arg_neuron_array[neurIt].transfer_function()
          );
          weightIt++;
        }

        /* Add recurrence of the Neuron */
        if(network_recurrence_to_self == recurrence){ /* recurrence to self */
          weight_synapse.set_interval_size(weight_synapse.interval_size() + 1 + 1); /* Update the weight synapse size: self-recurrence + a bias */

          /* Add the weight */
          arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
            arg_neuron_array[neurIt].transfer_function()
          );
          weightIt++;

          /* Add the input synapse */
          temp_input_interval.set_starts(neurIt); /* self-recurrence, an additional input snypse */
          temp_input_interval.set_interval_size(1); /* of a lone input as the actual @Neuron itself */
          temp_input_interval.set_reach_past_loops(1);
          *arg_neuron_array[neurIt].add_input_indices() = temp_input_interval;
        }else if(network_recurrence_to_layer == recurrence){ /* recurrence to layer */
          weight_synapse.set_interval_size(weight_synapse.interval_size() + layer_sizes[layer_index] + 1); /* Update the weight synapse size: Current layer + a bias */

          /* Add the weights */
          for(uint32 neuron_weight_iterator = 0; neuron_weight_iterator < layer_sizes[layer_index]; neuron_weight_iterator++){
            arg_weight_table[weightIt] = arg_weight_initer->next_weight_for(
              arg_neuron_array[neurIt].transfer_function()
            );
            weightIt++;
          }

          /* Add the input synapse */
          if(0 < layer_index)temp_input_interval.set_starts(layer_input_starts_at + layer_sizes[layer_index-1]); /* starts at the beginning of the current layer */
          else temp_input_interval.set_starts(layer_input_starts_at); /* starts at the beginning of the current layer */

          temp_input_interval.set_interval_size(layer_sizes[layer_index]); /* takes up the whole layer */
          temp_input_interval.set_reach_past_loops(1);
          *arg_neuron_array[neurIt].add_input_indices() = temp_input_interval;
        }else{ /* Only bias */
          weight_synapse.set_interval_size(weight_synapse.interval_size()+1); /* Update the weight synapse size: 1 bias */
        }
        arg_weight_table[weightIt] = arg_weight_initer->next_bias();
        weightIt++;
        *arg_neuron_array[neurIt].add_input_weights() = weight_synapse;
        neurIt++; /* Step the neuron iterator forward */
      }

      if(0 == layer_index){
        expPrevLayerOutput = arg_expected_input_range;
        layer_input_starts_at = 0;
      }else{
        expPrevLayerOutput /= static_cast<sdouble32>(layer_sizes[layer_index]);
        layer_input_starts_at += previous_size;
      }
      previous_size = layer_sizes[layer_index];
    } /* Iterate through all of the layers */

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
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(settings.get_arena_ptr());
    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(arg_output_neuron_number);
    set_weight_table(ret);
    set_neuron_array(ret);
    return ret;
  }else throw std::runtime_error("Inconsistent parameters given to Sparse Net Builder!");
}

} /* namespace rafko_net */
