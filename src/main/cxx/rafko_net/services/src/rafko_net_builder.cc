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

#include "rafko_net/services/rafko_net_builder.hpp"

#include <time.h>

#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/models/dense_net_weight_initializer.hpp"

namespace {

/* https://stackoverflow.com/questions/37563960/what-is-the-best-simplest-fastest-way-to-create-set-of-1-element-c */
template<typename Tp> inline std::set<Tp> make_set(Tp const&x){
  return {x};
}

} /* namespace */

namespace rafko_net {

RafkoNetBuilder& RafkoNetBuilder::add_feature_to_layer(std::uint32_t layer_index, Neuron_group_features feature){
  if(layer_features.find(layer_index) == layer_features.end()){
    layer_features[layer_index] = make_set(feature);
  }else{
    layer_features[layer_index].insert(feature);
  }
  return *this;
}

RafkoNet* RafkoNetBuilder::dense_layers(std::vector<std::uint32_t> layer_sizes){
  std::uint32_t previous_size = 0;
  std::uint32_t reach_back_max = 0;

  std::sort(arg_neuron_index_input_functions.begin(),arg_neuron_index_input_functions.end(),
  [](const std::tuple<std::uint32_t,std::uint32_t,Input_functions>& a, std::tuple<std::uint32_t,std::uint32_t,Input_functions>& b){
    return std::get<0>(a) < std::get<0>(b);
  });
  std::sort(arg_neuron_index_spike_functions.begin(),arg_neuron_index_spike_functions.end(),
  [](const std::tuple<std::uint32_t,std::uint32_t,Spike_functions>& a, std::tuple<std::uint32_t,std::uint32_t,Spike_functions>& b){
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

  if( /* Required arguments are set */
    (is_input_size_set && is_expected_input_range_set)
    &&(
      (!is_output_neuron_number_set) /* Output size is either not set */
      ||(arg_output_neuron_number == layer_sizes.back()) /* Or compliant to the Dense layer */
    )
  ){
    std::uint32_t layer_input_starts_at = 0;
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(settings.get_arena_ptr());
    double expPrevLayerOutput = TransferFunction::get_average_output_range(transfer_function_identity);

    ret->set_input_data_size(arg_input_size);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = arg_input_size;

    if(!is_weight_initializer_set)
      weight_initializer(std::make_shared<DenseNetWeightInitializer>(settings));

    std::uint32_t neuron_number_estimation = std::accumulate(layer_sizes.begin(), layer_sizes.end(), 0.0);
    arg_weight_table = std::vector<double>();
    arg_neuron_array = std::vector<Neuron>();
    arg_neuron_array.reserve(neuron_number_estimation);
    arg_weight_table.reserve(neuron_number_estimation * neuron_number_estimation);

    previous_size = arg_input_size;
    for(std::uint32_t layer_index = 0; layer_index < layer_sizes.size(); layer_index++)
    { /* Create the Dense Layers */
      if(0 == layer_sizes[layer_index]) throw std::runtime_error("Unable to construct zero sized layer!");
      invalidate(arg_neuron_index_input_functions, layer_index);
      invalidate(arg_neuron_index_spike_functions, layer_index);
      invalidate(arg_neuron_index_recurrence, layer_index);

      /* Configuring the weight_initializerializer for this layer */
      arg_weight_initer->set((0 == layer_index)?(arg_input_size):(layer_sizes[layer_index-1]),(expPrevLayerOutput));

      /* Store the features for this layer */
      bool layer_is_boltzmann_knot = false;
      if( layer_features.find(layer_index) != layer_features.end() ){
        for(const Neuron_group_features& feature_of_layer : layer_features[layer_index]){
          layer_is_boltzmann_knot = (neuron_group_feature_boltzmann_knot == feature_of_layer);
          RFASSERT_LOG(
            "Storing feature {} for the network starting from Neuron[{}]",
            Neuron_group_features_Name(feature_of_layer),
            (0u < layer_index)?(layer_input_starts_at + layer_sizes[layer_index - 1]):(0u)
          );
          FeatureGroup& feature_to_add = *ret->add_neuron_group_features();
          feature_to_add.set_feature(feature_of_layer);
          feature_to_add.add_relevant_neurons()->set_interval_size(layer_sizes[layer_index]);

          if(0u < layer_index)
            feature_to_add.mutable_relevant_neurons(0)->set_starts( layer_input_starts_at + layer_sizes[layer_index - 1] );
            else feature_to_add.mutable_relevant_neurons(0)->set_starts(0u);
        }
        layer_features.erase(layer_index);
      }

      /* Sort the input function requests based on Neuron indices this time */
      sort_next_layer(arg_neuron_index_input_functions, layer_index);
      sort_next_layer(arg_neuron_index_spike_functions, layer_index);
      sort_next_layer(arg_neuron_index_recurrence, layer_index);

      /* Add the Neurons */
      expPrevLayerOutput = 0;
      for(std::uint32_t layer_neuron_index = 0; layer_neuron_index < layer_sizes[layer_index]; layer_neuron_index++){
        arg_neuron_array.push_back(Neuron());
        std::optional<Input_functions> neuron_input_function = get_value(
          arg_neuron_index_input_functions, layer_neuron_index
        );
        if(neuron_input_function.has_value())
          arg_neuron_array.back().set_input_function( InputFunction::next({neuron_input_function.value()}) );
          else arg_neuron_array.back().set_input_function(InputFunction::next());
        invalidate(arg_neuron_index_input_functions, layer_index, layer_neuron_index);

        if(is_allowed_transfer_functions_by_layer_set)
          arg_neuron_array.back().set_transfer_function(TransferFunction::next(arg_allowed_transfer_functions_by_layer[layer_index]));
        else arg_neuron_array.back().set_transfer_function(TransferFunction::next());

        std::optional<Spike_functions> neuron_spike_function = get_value(
          arg_neuron_index_spike_functions, layer_neuron_index
        );
        if(neuron_spike_function.has_value())
          arg_neuron_array.back().set_spike_function(SpikeFunction::next({neuron_spike_function.value()}));
          else arg_neuron_array.back().set_spike_function(SpikeFunction::next());
        invalidate(arg_neuron_index_spike_functions, layer_index, layer_neuron_index);

        /* Storing the expected output of this Net */
        if(0 < layer_index)expPrevLayerOutput += TransferFunction::get_average_output_range(
          arg_neuron_array.back().transfer_function()
        );

        std::uint32_t input_weights_to_add = previous_size; /* starting value is the previous layer  inputs ( below ) */
        { /* Add the previous layer as an input */
          InputSynapseInterval& interval = *arg_neuron_array.back().add_input_indices();
          if(0 == layer_index){
            interval.set_starts(SynapseIterator<>::external_index_from_array_index(0));
          }else{
            interval.set_starts(layer_input_starts_at);
          }
          interval.set_interval_size(previous_size);
          interval.set_reach_past_loops(0);
        }

        if(layer_is_boltzmann_knot){ /* recurrence to layer */
            InputSynapseInterval& interval = *arg_neuron_array.back().add_input_indices();
            if(0 < layer_index)
              interval.set_starts(layer_input_starts_at + layer_sizes[layer_index - 1]); /* starts at the beginning of the current layer */
              else interval.set_starts(layer_input_starts_at); /* starts at the beginning of the current layer */
            interval.set_interval_size(layer_sizes[layer_index]); /* takes up the whole layer */
            interval.set_reach_past_loops(1u);
            reach_back_max = std::max(reach_back_max, 1u);
            input_weights_to_add += layer_sizes[layer_index];
        }

        /* Add self-recurrences to Neuron */
        std::optional<std::uint32_t> past_index = get_value(arg_neuron_index_recurrence, layer_neuron_index);
        while(past_index.has_value()){ /* recurrence to self */
          ++input_weights_to_add;

          /* Add the input synapse */
          InputSynapseInterval& interval = *arg_neuron_array.back().add_input_indices();
          interval.set_starts(layer_input_starts_at + layer_neuron_index); /* self-recurrence, an additional input snypse */
          interval.set_interval_size(1u); /* of a lone input as the actual @Neuron itself */
          interval.set_reach_past_loops(past_index.value());
          reach_back_max = std::max(reach_back_max, past_index.value());
          /* pop the current and try to get the next recurrent value */
          arg_neuron_index_recurrence.erase(arg_neuron_index_recurrence.begin());
          past_index = get_value(arg_neuron_index_recurrence, layer_neuron_index);
        }

        /* Add the weights for the previous layer to the built net */
        IndexSynapseInterval& weight_synapse = *arg_neuron_array.back().add_input_weights();
        weight_synapse.set_starts(arg_weight_table.size());
        weight_synapse.set_interval_size(input_weights_to_add + 2u); /* input weights plus bias plus the spike weight */
        arg_weight_table.push_back(arg_weight_initer->next_memory_filter()); /* Add the spike function weight as the first weight of the Neuron */
        arg_weight_table.resize(arg_weight_table.size() + input_weights_to_add); /* Add the weight for all the inputs */
        std::generate(arg_weight_table.end() - input_weights_to_add, arg_weight_table.end(), [this](){
          return arg_weight_initer->next_weight_for( arg_neuron_array.back().transfer_function() );
        });
        arg_weight_table.push_back(arg_weight_initer->next_bias()); /* Add bias weight */
      }/*for(neurons inside the layer)*/

      if(0 == layer_index){
        expPrevLayerOutput = arg_expected_input_range;
      }else{
        expPrevLayerOutput /= static_cast<double>(layer_sizes[layer_index]);
        layer_input_starts_at += previous_size;
      }
      previous_size = layer_sizes[layer_index];
    } /* Iterate through all of the layers */

    set_weight_table(ret);
    set_neuron_array(ret);
    ret->set_memory_size(reach_back_max + 1u);
    return ret;
  }else throw std::runtime_error("Input Output Pre-requisites failed;Unable to determine Net Structure!");
}

} /* namespace rafko_net */
