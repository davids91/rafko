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

#ifndef RAFKO_NET_BUILDER_H
#define RAFKO_NET_BUILDER_H

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <set>
#include <optional>
#include <unordered_map>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_net/models/spike_function.hpp"
#include "rafko_net/models/weight_initializer.hpp"
#include "rafko_net/models/neuron_info.hpp"

namespace rafko_net {

/**
 * @brief RafkoNetBuilder: Builder class to compile Sparse Neural Networks
 * There are Two ways to use this class. One is to add the required building blocks of a Network
 * manually. The Other is to use one of the higher level construction functions like @RafkoNetBuilder::dense_layers.
 * Some parameters needed to be added unconditionally, which is checked by @RafkoNetBuilder::io_pre_requisites_set.
 */
class RAFKO_FULL_EXPORT RafkoNetBuilder{
public:
  RafkoNetBuilder(const rafko_mainframe::RafkoSettings& settings_)
  :  settings(settings_)
  { }

  /**
   * @brief      RafkoNetBuilder::input_size: sets the number of expected inputs for the RafkoNet object to be built
   *
   * @param[in]  size  The size
   *
   * @return     builder reference for chaining
   */
  constexpr RafkoNetBuilder& input_size(std::uint32_t size){
    arg_input_size = size;
    is_input_size_set = true;
    return *this;
  }

  /**
   * @brief      sets the number of expected outputs for the RafkoNet object to be built
   *
   * @param[in]  size  The size
   *
   * @return     builder reference for chaining
   */
  constexpr RafkoNetBuilder& output_neuron_number(std::uint32_t size){
    arg_output_neuron_number = size;
    is_output_neuron_number_set = true;
    return *this;
  }

  /**
   * @brief      Sets the expected range of inputs to the net
   *
   * @param[in]  range  The range
   *
   * @return
   */
  constexpr RafkoNetBuilder& expected_input_range(double range){
    arg_expected_input_range = range;
    is_expected_input_range_set = true;
    return *this;
  }

  /**
   * @brief      Sets the Weight initializer to a manual one, overwriting the default weight
   *             intialization assigned for any builder interface, except @RafkoNetBuilder::build.
   *
   * @param[in]  initializer  The initializer
   *
   * @return     Builder reference for chaining
   */
  RafkoNetBuilder& weight_initializer(std::shared_ptr<WeightInitializer> initializer){
    if(nullptr != initializer){
      arg_weight_initer = initializer;
      is_weight_initializer_set = true;
    }else is_weight_initializer_set = false;
    return *this;
  }

  /**
   * @brief      set the given neuron_array and transfer its ownership to the builder
   *
   * @param[in]  arr   The array of neurons to be transferred
   *
   * @return     Builder reference for chaining
   */
  RafkoNetBuilder& neuron_array(std::vector<Neuron> arr){
    if((0 < arr.size())&&(NeuronInfo::is_neuron_valid(arr.back()))){
      arg_neuron_array = arr;
      is_neuron_array_set = true;
    }else is_neuron_array_set = false;
    return *this;
  }

  /**
   * @brief      set the given weight table and transfer ownership to the builder
   *
   * @param[in]  table  The table to be transferred
   *
   * @return     reference for chaining
   */
  RafkoNetBuilder& weight_table(std::vector<double> table){
    if(0 < table.size()){
      arg_weight_table = table;
      is_weight_table_set = true;
    }else is_weight_table_set = false;
    return *this;
  }

  /**
   * @brief      Sets an optional argument which restricts transfer functions by layer ( usable with @dense_layers )
   *
   * @param[in]  allowed_transfer_functions_by_layer  The allowed transfer functions by layer
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& allowed_transfer_functions_by_layer(std::vector<std::set<Transfer_functions> > filter){
    arg_allowed_transfer_functions_by_layer = filter;
    is_allowed_transfer_functions_by_layer_set = true;
    return *this;
  }

  /**
   * @brief      Add a feature to the layer of the network to be built
   *
   * @param[in]   layer_index   The index of the Layer to set the features on
   * @param[in]   feature       The feature to set to the layer
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& add_feature_to_layer(std::uint32_t layer_index, Neuron_group_features feature);

  /**
   * @brief      Set the input function of a Neuron other, than the default "+". Overwrites other input function that might be set for this exact Neuron.
   *
   * @param[in]   layer_index     The index of the Layer to set the features on
   * @param[in]   neuron_index    The relative index of the neuron inside the layer
   * @param[in]   function        The function to set to the Neuron
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& set_neuron_input_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Input_functions function);

  /**
   * @brief      Set the transfer function of a Neuron Explicitly. Overwrites other input function that might be set for this exact Neuron.
   *
   * @param[in]   layer_index     The index of the Layer to set the features on
   * @param[in]   neuron_index    The relative index of the neuron inside the layer
   * @param[in]   function        The function to set to the Neuron
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& set_neuron_transfer_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Transfer_functions function);

  /**
   * @brief      Set the spike function of a Neuron other, than the default @spike_function_memory.  Overwrites other input function that might be set for this exact Neuron.
   *
   * @param[in]   layer_index     The index of the Layer to set the features on
   * @param[in]   neuron_index    The relative index of the neuron inside the layer
   * @param[in]   function        The function to set to the Neuron
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& set_neuron_spike_function(std::uint32_t layer_index, std::uint32_t layer_neuron_index, Spike_functions function);

  /**
   * @brief      Makes the Neuron take input from itself in the previous run
   *
   * @param[in]   layer_index     The index of the Layer to set the features on
   * @param[in]   neuron_index    The relative index of the neuron inside the layer
   * @param[in]   feature         The function to set to the Neuron
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& add_neuron_recurrence(std::uint32_t layer_index, std::uint32_t layer_neuron_index, std::uint32_t past){
    arg_neuron_index_recurrence.push_back({layer_index, layer_neuron_index, past});
    return *this;
  }

  /**
   * @brief      creates a Fully connected feedforward neural network based on the IO arguments and
   *             and function arguments. The structure is according to the provided layer sizes
   *             argument, where the neurons of a layer is after in the previous layers, and before
   *             the succeeding layer Neurons.
   *
   * @param[in]  layerSizes                 how many layers will there be in the result and how big are those layers going to be
   * @param[in]  transfer_function_filter   The allowed transfer functions per layer
   *
   * @return   the built neural network
   */
  RafkoNet* dense_layers(std::vector<std::uint32_t> layer_sizes, std::vector<std::set<Transfer_functions>> transfer_function_filter){
    (void)allowed_transfer_functions_by_layer(transfer_function_filter);
    return dense_layers(layer_sizes);
  }

  /**
   * @brief      Same as above, but without any Transfer function restrictions
   *
   * @param[in]  layer_sizes  The layer sizes
   *
   * @return     the built neural network
   */
  RafkoNet* dense_layers(std::vector<std::uint32_t> layer_sizes);

private:
  const rafko_mainframe::RafkoSettings& settings;

  /**
   * @brief   Helper variables to see if different required arguments are set inside the builder
   */
  bool is_input_size_set = false;
  bool is_output_neuron_number_set = false;
  bool is_expected_input_range_set = false;
  bool is_weight_table_set = false;
  bool is_weight_initializer_set = false;
  bool is_neuron_array_set = false;
  bool is_allowed_transfer_functions_by_layer_set = false;

  /**
   * @brief   Helper variables for features and optional Neuron parameters
   */
  std::vector< std::set<Transfer_functions> > arg_allowed_transfer_functions_by_layer;
  std::unordered_map<std::uint32_t, std::set<Neuron_group_features>> layer_features;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Input_functions> > arg_neuron_index_input_functions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Transfer_functions> > arg_neuron_index_transfer_functions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Spike_functions> > arg_neuron_index_spike_functions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,std::uint32_t> > arg_neuron_index_recurrence;

  /**
   * The absolute value of the amplitude of one average input datapoint. It supports weight initialization.
   */
  double arg_expected_input_range = TransferFunction::get_average_output_range(transfer_function_identity);

  /**
   * The array containing the neurons while RafkoNetBuilder::build is used
   */
  std::vector<Neuron> arg_neuron_array;

  /**
   * The array containing the used weights in the network while RafkoNetBuilder::build is used
   */
  std::vector<double> arg_weight_table;

  /**
   * Weight Initializer argument, which guides the initial net Weights
   */
  std::shared_ptr<WeightInitializer> arg_weight_initer;

  /**
   * Number of inputs the net-to-be-built shall accept
   */
  std::uint32_t arg_input_size = 0;

  /**
   * Number of Neurons the net-to-be-built shall have as output
   */
  std::uint32_t arg_output_neuron_number = 0;

};

} /* namespace rafko_net */
#endif /* RAFKO_NET_BUILDER_H */