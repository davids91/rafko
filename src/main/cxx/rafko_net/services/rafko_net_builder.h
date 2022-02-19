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

#include "rafko_global.h"

#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/models/transfer_function.h"
#include "rafko_net/models/weight_initializer.h"
#include "rafko_net/models/neuron_info.h"

namespace rafko_net {

/**
 * @brief RafkoNetBuilder: Builder class to compile Sparse Neural Networks
 * There are Two ways to use this class. One is to add the required building blocks of a Network
 * manually. The Other is to use one of the higher level construction functions like @RafkoNetBuilder::dense_layers.
 * Some parameters needed to be added unconditionally, which is checked by @RafkoNetBuilder::io_pre_requisites_set.
 */
class RAFKO_FULL_EXPORT RafkoNetBuilder{
public:
  RafkoNetBuilder(rafko_mainframe::RafkoSettings& settings_)
  :  settings(settings_)
  { }

  /**
   * @brief      RafkoNetBuilder::input_size: sets the number of expected inputs for the RafkoNet object to be built
   *
   * @param[in]  size  The size
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& input_size(uint32 size){
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
  RafkoNetBuilder& output_neuron_number(uint32 size){
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
  RafkoNetBuilder& expected_input_range(sdouble32 range){
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
  RafkoNetBuilder& weight_table(std::vector<sdouble32> table){
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
  RafkoNetBuilder& allowed_transfer_functions_by_layer(std::vector<std::vector<Transfer_functions> > filter){
    arg_allowed_transfer_functions_by_layer = filter;
    is_allowed_transfer_functions_by_layer_set = true;
    return *this;
  }

  /**
   * @brief      If supported, produced network will also contain for every @Neuron
   *             its previous input, in case this function is called
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& set_recurrence_to_self(){
    recurrence = network_recurrence_to_self;
    return *this;
  }

  /**
   * @brief      If supported, produced network will also contain for every layer
   *             the activation of the layer from the previous run
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& set_recurrence_to_layer(){
    recurrence = network_recurrence_to_layer;
    return *this;
  }

  /**
   * @brief      If supported, produced network will also contain for every layer
   *             the activation of the layer from the previous run
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& add_feature_to_layer(uint32 layer_index, Neuron_group_features feature);

  /**
   * @brief      creates a Fully connected feedforward neural network based on the IO arguments and
   *             and function arguments. The structure is according to the provided layer sizes
   *             argument, where the neurons of a layer is after in the previous layers, and before
   *             the succeeding layer Neurons.
   *
   * @param[in]  layerSizes         how many layers will there be in the result
   *                    and how big are those layers going to be
   * @param[in]  transfer_function_filter  The allowed transfer functions per layer
   *
   * @return   the built neural network
   */
  RafkoNet* dense_layers(std::vector<uint32> layer_sizes, std::vector<std::vector<Transfer_functions>> transfer_function_filter){
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
  RafkoNet* dense_layers(std::vector<uint32> layer_sizes);

  /**
   * @brief    creates a Neural network from the given Arguments. Requires the following
   *       components to be set:
   *       - The Neuron Array contains in an array the initialized Neuron
   *       - The Weight Table containing the weights used by the Neural network
   *       Building Networks like this is very dangerous, as the integrity of the components are not
   *       checked, therefore the user of this interface should be responsible for the behavior of the
   *       resulted Neural Network.
   *
   * @return   the built neural network
   */
  RafkoNet* build();

private:
  rafko_mainframe::RafkoSettings& settings;

  /**
   * Helper variables to see if different required arguments are set inside the builder
   */
  bool is_input_size_set = false;
  bool is_output_neuron_number_set = false;
  bool is_expected_input_range_set = false;
  bool is_weight_table_set = false;
  bool is_weight_initializer_set = false;
  bool is_neuron_array_set = false;
  bool is_allowed_transfer_functions_by_layer_set = false;
  uint32 recurrence = network_recurrence_unknown;

  /**
   * The absolute value of the amplitude of one average input datapoint. It supports weight initialization.
   */
  sdouble32 arg_expected_input_range = TransferFunction::get_average_output_range(transfer_function_identity);

  /**
   * The array containing the neurons while RafkoNetBuilder::build is used
   */
  std::vector<Neuron> arg_neuron_array;

  /**
   * The array containing the used weights in the network while RafkoNetBuilder::build is used
   */
  std::vector<sdouble32> arg_weight_table;

  /**
   * Weight Initializer argument, which guides the initial net Weights
   */
  std::shared_ptr<WeightInitializer> arg_weight_initer;

  /**
   * Number of inputs the net-to-be-built shall accept
   */
  uint32 arg_input_size = 0;

  /**
   * Number of Neurons the net-to-be-built shall have as output
   */
  uint32 arg_output_neuron_number = 0;

  std::vector<std::vector<Transfer_functions> > arg_allowed_transfer_functions_by_layer;
  std::vector<std::pair<uint32,Neuron_group_features>> layer_features;

  /**
   * @brief RafkoNetBuilder::set_neuron_array: moves the neuron_array argument into the RafkoNet
   * @param arr: the neuron array to be added to the @RafkoNet object net
   * @param net: the new owner of the neuron_array
   */
  void set_neuron_array(RafkoNet* net){
    if(NeuronInfo::is_neuron_valid(arg_neuron_array.back())){ /* If the last element is valid */
      *net->mutable_neuron_array() = {arg_neuron_array.begin(),arg_neuron_array.end()};
    } else throw std::runtime_error("Unable to set Neuron Array into Sparse net as the last Neuron seems invalid!");
  }

  /**
   * @brief RafkoNetBuilder::set_weight_table: moves the weightTable argument into the RafkoNet
   * @param table: the array of floating point numbers to be added to the @RafkoNet object net
   * @param net: the new owner of the weightTable
   */
  void set_weight_table(RafkoNet* net){
    if(0 < arg_weight_table.size()){
      *net->mutable_weight_table() = {arg_weight_table.begin(), arg_weight_table.end()};
    }else throw std::runtime_error("Unable to build net, weight table is of size 0!");
  }

};

} /* namespace rafko_net */
#endif /* RAFKO_NET_BUILDER_H */
