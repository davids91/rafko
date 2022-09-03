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

/**
 * @brief     Inserts the provided arguments into the feature vector
 *
 * @param         feature               A vector of {{layer_index,neuron_index},T} pairs, where T is the feature the builder stores
 * @param[in]     layer_index           The layer index determines which elements inside the feature are deprecated and to be removed
 * @param[in]     layer_neuron_index    The layer index determines which elements inside the feature are deprecated and to be removed
 */
template<typename T>
static void insert_into(
  std::vector< std::tuple<std::uint32_t,std::uint32_t,T> >& feature,
  std::uint32_t layer_index, std::uint32_t layer_neuron_index, T function
){
  typename std::vector<std::tuple<std::uint32_t,std::uint32_t,T>>::iterator
  found_element = std::find_if(feature.begin(), feature.end(),
    [layer_index,layer_neuron_index](const std::tuple<std::uint32_t,std::uint32_t,T>& current_element){
      return ( (layer_index == std::get<0>(current_element))&&(layer_neuron_index == std::get<1>(current_element)) );
    }
  );

  if(found_element == feature.end())
    feature.push_back({layer_index, layer_neuron_index,function});
    else *found_element = {layer_index, layer_neuron_index, function};
}

/**
 * @brief     Looks inside the provided vector and removes the irrelevant parts based on the given layer index
 *            The function works assuming that the vector is sorted by the layer index in ascending order
 *
 * @param         feature                 A vector of {{layer_index,layer_neuron_index},T} pairs, where T is the feature the builder stores
 * @param[in]     current_layer_index     The layer index determines which elements inside the feature are deprecated and to be removed
 */
template<typename T>
static void invalidate(
  std::vector< std::tuple<std::uint32_t,std::uint32_t,T> >& feature,
  std::uint32_t current_layer_index
){ /* erase all the deprecated input function settings for Neurons based on the layer index */
  while( (0u < feature.size())&&(current_layer_index > std::get<0>(feature.front())) )
    feature.erase(feature.begin());
}

/**
 * @brief     Looks inside the provided vector and removes the irrelevant parts based on the given layer index
 *            Assumes that the vector is sorted by the layer index in ascending order;
 *            and the elements are also sorted in ascending order by neuron index per layer!
 *
 * @param         feature                 A vector of {{layer_index,neuron_index},T} pairs, where T is the feature the builder stores
 * @param[in]     current_layer_index     The layer index determines which elements inside the feature are deprecated and to be removed
 * @param[in]     layer_neuron_index      The layer index determines which elements inside the feature are deprecated and to be removed
 */
template<typename T>
static void invalidate(
  std::vector< std::tuple<std::uint32_t,std::uint32_t,T> >& feature,
  std::uint32_t current_layer_index, std::uint32_t layer_neuron_index
){
  while( /* erase all the deprecated input function settings for Neurons based on Neuron index inside the layer */
    ( 0u < feature.size() )
    &&(current_layer_index == std::get<0>(feature.front()))
    &&(layer_neuron_index >= std::get<1>(feature.front()))
  )feature.erase(feature.begin());
}

/**
 * @brief     Sorts part the given feature vector for neuron index values.
 *            The part which is sorted is determined by the provided layer_index
 *
 * @param         feature                 A vector of {{layer_index,layer_neuron_index},T} pairs, where T is the feature the builder stores
 * @param[in]     current_layer_index     The layer index determines which elements inside the feature are to be sorted0
 */
template<typename T>
static void sort_next_layer(
  std::vector< std::tuple<std::uint32_t,std::uint32_t,T> >& feature,
  std::uint32_t current_layer_index
){
  std::sort( /* starting from the beginning of the array.. */
    feature.begin(), /* ..because builder continually removes the irrelevant front parts.. */
    std::find_if(feature.begin(),feature.end(), /* .. so the first index for the next layer is the end of the part we need to sort */
      [current_layer_index](const std::tuple<std::uint32_t,std::uint32_t,T>& element){
        return std::get<0>(element) == (current_layer_index + 1u); /* ..until the part of the array which starts at the next layer. */
      }
      /*!Note: this works because the vector is already sorted based on layer index,
       * so the element in this `find_if` is either the end of the vector
       * or the start of the next layer relevant part of it
       */
    ),
    [](const std::tuple<std::uint32_t,std::uint32_t,T>& a, std::tuple<std::uint32_t,std::uint32_t,T>& b){
      return std::get<1>(a) < std::get<1>(b); /* the relevant(to the layer_index)y part of the vector is sorted based on Neuron index */
    }
  );
}

/**
 * @brief     Provides the value of the feature mapped to the neuron mapped inside the given layer
 *            if it is set by the provided feature vector.
 *            Assumes that the vector is sorted by the layer index in ascending order;
 *            and the elements are also sorted in ascending order by neuron index per layer!
 *            Also assumes that the feature is available to the given layer_neuron_index
 *            if, and only if `feature.front()` contains the index.
 *
 * @param         feature                 A vector of {{layer_index,layer_neuron_index},T} pairs, where T is the feature the builder stores
 * @param[in]     current_layer_index     The layer index determines which elements inside the feature are deprecated and to be removed
 * @param[in]     layer_neuron_index      The layer index determines which elements inside the feature are deprecated and to be removed
 * @returns       The feature T, if set.
 */
template<typename T>
static std::optional<T> get_value(
  std::vector< std::tuple<std::uint32_t,std::uint32_t,T> >& feature,
  std::uint32_t layer_neuron_index
){  /* if a Neuron has its feature explicitly set */
  if( (0u < feature.size())&&(layer_neuron_index == std::get<1>(feature.front())) )
    return std::get<2>(feature.front());
    else return {};
}

} /* namespace */

namespace rafko_net {

RafkoNetBuilder& RafkoNetBuilder::add_feature_to_layer(std::uint32_t layer_index, Neuron_group_features feature){
  if(m_layerFeatures.find(layer_index) == m_layerFeatures.end()){
    m_layerFeatures[layer_index] = make_set(feature);
  }else{
    m_layerFeatures[layer_index].insert(feature);
  }
  return *this;
}

RafkoNetBuilder& RafkoNetBuilder::set_neuron_input_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Input_functions function){
  insert_into(m_argNeuronIndexInputFunctions, layer_index, layer_neuron_index, function);
  return *this;
}

RafkoNetBuilder& RafkoNetBuilder::set_neuron_transfer_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Transfer_functions function){
  insert_into(m_argNeuronIndexTransferFunctions, layer_index, layer_neuron_index, function);
  return *this;
}

RafkoNetBuilder& RafkoNetBuilder::set_neuron_spike_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Spike_functions function){
  insert_into(m_argNeuronIndexSpikeFunctions, layer_index, layer_neuron_index, function);
  return *this;
}

void RafkoNetBuilder::build_dense_layers_and_update(
  RafkoNet* previous, std::vector<std::uint32_t> layer_sizes,
  std::vector<std::set<Transfer_functions>> transfer_function_filter
){
  RFASSERT(nullptr != previous);
  RafkoNet* built_network = dense_layers(nullptr, layer_sizes, transfer_function_filter);
  built_network->Swap(previous);
  delete built_network;
}


RafkoNet* RafkoNetBuilder::dense_layers(google::protobuf::Arena* arena_ptr, std::vector<std::uint32_t> layer_sizes, std::vector<std::set<Transfer_functions>> transfer_function_filter){
  std::uint32_t previous_size = 0;
  std::uint32_t reach_back_max = 0;

  if(transfer_function_filter.size() == layer_sizes.size()){
    (void)allowed_transfer_functions_by_layer(transfer_function_filter);
  }

  std::sort(m_argNeuronIndexInputFunctions.begin(),m_argNeuronIndexInputFunctions.end(),
  [](const std::tuple<std::uint32_t,std::uint32_t,Input_functions>& a, std::tuple<std::uint32_t,std::uint32_t,Input_functions>& b){
    return std::get<0>(a) < std::get<0>(b);
  });
  std::sort(m_argNeuronIndexSpikeFunctions.begin(),m_argNeuronIndexSpikeFunctions.end(),
  [](const std::tuple<std::uint32_t,std::uint32_t,Spike_functions>& a, std::tuple<std::uint32_t,std::uint32_t,Spike_functions>& b){
    return std::get<0>(a) < std::get<0>(b);
  });

  if(
    (m_isAllowedTransferFunctionsByLayerSet)
    &&(layer_sizes.size() != m_argAllowedTransferFunctionsByLayer.size())
  )throw std::runtime_error(
    "Allowed transfer functions need to be set for each layer explicitly; sizes don't match: "
    + std::string("layers: ") + std::to_string(layer_sizes.size())
    + std::string(" vs ") + std::to_string(m_argAllowedTransferFunctionsByLayer.size())
  );

  if( /* Required arguments are set */
    (m_isInputSizeSet && m_isExpectedInputRangeSet)
    &&(
      (!m_isOutputNeuronNumberSet) /* Output size is either not set */
      ||(m_argOutputNeuronNumber == layer_sizes.back()) /* Or compliant to the Dense layer */
    )
  ){
    std::uint32_t layer_input_starts_at = 0;
    RafkoNet* ret = google::protobuf::Arena::CreateMessage<RafkoNet>(arena_ptr);
    double expPrevLayerOutput = TransferFunction::get_average_output_range(transfer_function_identity);

    ret->set_input_data_size(m_argInputSize);
    ret->set_output_neuron_number(layer_sizes.back());

    previous_size = m_argInputSize;

    if(!m_isWeightInitializerSet)
      weight_initializer(std::make_shared<DenseNetWeightInitializer>(m_settings));

    std::uint32_t neuron_number_estimation = std::accumulate(layer_sizes.begin(), layer_sizes.end(), 0.0);
    m_argWeightTable = std::vector<double>();
    m_argNeuronArray = std::vector<Neuron>();
    m_argNeuronArray.reserve(neuron_number_estimation);
    m_argWeightTable.reserve(neuron_number_estimation * neuron_number_estimation);

    previous_size = m_argInputSize;
    for(std::uint32_t layer_index = 0; layer_index < layer_sizes.size(); layer_index++)
    { /* Create the Dense Layers */
      if(0 == layer_sizes[layer_index]) throw std::runtime_error("Unable to construct zero sized layer!");
      invalidate(m_argNeuronIndexInputFunctions, layer_index);
      invalidate(m_argNeuronIndexSpikeFunctions, layer_index);
      invalidate(m_argNeuronIndexRecurrence, layer_index);

      /* Configuring the weight_initializerializer for this layer */
      m_argWeightIniter->set((0 == layer_index)?(m_argInputSize):(layer_sizes[layer_index-1]),(expPrevLayerOutput));

      /* Store the features for this layer */
      bool layer_is_boltzmann_knot = false;
      if( m_layerFeatures.find(layer_index) != m_layerFeatures.end() ){
        for(const Neuron_group_features& feature_of_layer : m_layerFeatures[layer_index]){
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
        m_layerFeatures.erase(layer_index);
      }

      /* Sort the input function requests based on Neuron indices this time */
      sort_next_layer(m_argNeuronIndexInputFunctions, layer_index);
      sort_next_layer(m_argNeuronIndexTransferFunctions, layer_index);
      sort_next_layer(m_argNeuronIndexSpikeFunctions, layer_index);
      sort_next_layer(m_argNeuronIndexRecurrence, layer_index);

      /* Add the Neurons */
      expPrevLayerOutput = 0;
      for(std::uint32_t layer_neuron_index = 0; layer_neuron_index < layer_sizes[layer_index]; layer_neuron_index++){
        m_argNeuronArray.push_back(Neuron());
        std::optional<Input_functions> neuron_input_function = get_value(
          m_argNeuronIndexInputFunctions, layer_neuron_index
        );
        if(neuron_input_function.has_value())
          m_argNeuronArray.back().set_input_function( InputFunction::next({neuron_input_function.value()}) );
          else m_argNeuronArray.back().set_input_function(InputFunction::next());
        invalidate(m_argNeuronIndexInputFunctions, layer_index, layer_neuron_index);

        std::optional<Transfer_functions> neuron_transfer_function = get_value(
          m_argNeuronIndexTransferFunctions, layer_neuron_index
        );
        if(neuron_transfer_function.has_value()){
          if(
            m_isAllowedTransferFunctionsByLayerSet
            &&(0u == m_argAllowedTransferFunctionsByLayer[layer_index].count(neuron_transfer_function.value()))
          ){
            throw std::runtime_error(
              "Layer[" +  std::to_string(layer_index) + "] Neuron[(in layer)" + std::to_string(layer_neuron_index) + "]"
              + "set transfer function conflicts with allowed Transfer functions per layer!"
            );
          }else{
            m_argNeuronArray.back().set_transfer_function(neuron_transfer_function.value());
          }
        }else if(m_isAllowedTransferFunctionsByLayerSet)
          m_argNeuronArray.back().set_transfer_function(TransferFunction::next(m_argAllowedTransferFunctionsByLayer[layer_index]));
        else m_argNeuronArray.back().set_transfer_function(TransferFunction::next());
        invalidate(m_argNeuronIndexTransferFunctions, layer_index, layer_neuron_index);

        std::optional<Spike_functions> neuron_spike_function = get_value(
          m_argNeuronIndexSpikeFunctions, layer_neuron_index
        );
        if(neuron_spike_function.has_value())
          m_argNeuronArray.back().set_spike_function(SpikeFunction::next({neuron_spike_function.value()}));
          else m_argNeuronArray.back().set_spike_function(SpikeFunction::next());
        invalidate(m_argNeuronIndexSpikeFunctions, layer_index, layer_neuron_index);

        /* Storing the expected output of this Net */
        if(0 < layer_index)expPrevLayerOutput += TransferFunction::get_average_output_range(
          m_argNeuronArray.back().transfer_function()
        );

        std::uint32_t input_weights_to_add = previous_size; /* starting value is the previous layer  inputs ( below ) */
        { /* Add the previous layer as an input */
          InputSynapseInterval& interval = *m_argNeuronArray.back().add_input_indices();
          if(0 == layer_index){
            interval.set_starts(SynapseIterator<>::external_index_from_array_index(0));
          }else{
            interval.set_starts(layer_input_starts_at);
          }
          interval.set_interval_size(previous_size);
          interval.set_reach_past_loops(0);
        }

        if(layer_is_boltzmann_knot){ /* recurrence to layer */
            InputSynapseInterval& interval = *m_argNeuronArray.back().add_input_indices();
            if(0 < layer_index)
              interval.set_starts(layer_input_starts_at + layer_sizes[layer_index - 1]); /* starts at the beginning of the current layer */
              else interval.set_starts(layer_input_starts_at); /* starts at the beginning of the current layer */
            interval.set_interval_size(layer_sizes[layer_index]); /* takes up the whole layer */
            interval.set_reach_past_loops(1u);
            reach_back_max = std::max(reach_back_max, 1u);
            input_weights_to_add += layer_sizes[layer_index];
        }

        /* Add self-recurrences to Neuron */
        std::optional<std::uint32_t> past_index = get_value(m_argNeuronIndexRecurrence, layer_neuron_index);
        while(past_index.has_value()){ /* recurrence to self */
          ++input_weights_to_add;

          /* Add the input synapse */
          InputSynapseInterval& interval = *m_argNeuronArray.back().add_input_indices();
          interval.set_starts(layer_input_starts_at + layer_neuron_index); /* self-recurrence, an additional input snypse */
          interval.set_interval_size(1u); /* of a lone input as the actual @Neuron itself */
          interval.set_reach_past_loops(past_index.value());
          reach_back_max = std::max(reach_back_max, past_index.value());
          /* pop the current and try to get the next recurrent value */
          m_argNeuronIndexRecurrence.erase(m_argNeuronIndexRecurrence.begin());
          past_index = get_value(m_argNeuronIndexRecurrence, layer_neuron_index);
        }

        /* Add the weights for the previous layer to the built net */
        IndexSynapseInterval& weight_synapse = *m_argNeuronArray.back().add_input_weights();
        weight_synapse.set_starts(m_argWeightTable.size());
        weight_synapse.set_interval_size(input_weights_to_add + 2u); /* input weights plus bias plus the spike weight */
        m_argWeightTable.push_back(m_argWeightIniter->next_memory_filter()); /* Add the spike function weight as the first weight of the Neuron */
        m_argWeightTable.resize(m_argWeightTable.size() + input_weights_to_add); /* Add the weight for all the inputs */
        std::generate(m_argWeightTable.end() - input_weights_to_add, m_argWeightTable.end(), [this](){
          return m_argWeightIniter->next_weight_for( m_argNeuronArray.back().transfer_function() );
        });
        m_argWeightTable.push_back(m_argWeightIniter->next_bias()); /* Add bias weight */
      }/*for(neurons inside the layer)*/

      if(0 == layer_index){
        expPrevLayerOutput = m_argExpectedInputRange;
      }else{
        expPrevLayerOutput /= static_cast<double>(layer_sizes[layer_index]);
        layer_input_starts_at += previous_size;
      }
      previous_size = layer_sizes[layer_index];
    } /* Iterate through all of the layers */

    if(0 < m_argWeightTable.size()){
      *ret->mutable_weight_table() = {m_argWeightTable.begin(), m_argWeightTable.end()};
    }else throw std::runtime_error("Unable to build net, weight table is of size 0!");

    if(NeuronInfo::is_neuron_valid(m_argNeuronArray.back())){ /* If the last element is valid */
      *ret->mutable_neuron_array() = {m_argNeuronArray.begin(),m_argNeuronArray.end()};
    }else throw std::runtime_error("Unable to set Neuron Array into Sparse net as the last Neuron seems invalid!");

    ret->set_memory_size(reach_back_max + 1u);
    return ret;
  }else throw std::runtime_error("Input Output Pre-requisites failed;Unable to determine Net Structure!");
}

} /* namespace rafko_net */
