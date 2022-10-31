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
#include "rafko_utilities/services/rafko_math_utils.hpp"
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_net/models/spike_function.hpp"
#include "rafko_net/models/weight_initializer.hpp"
#include "rafko_net/models/neuron_info.hpp"

namespace rafko_net {

/**
 * @brief RafkoNetBuilder: Builder class to compile Sparse Neural Networks
 * There are Two ways to use this class. One is to add the required building blocks of a Network
 * manually. The Other is to use one of the higher level construction functions like @RafkoNetBuilder::create_layers.
 * Some parameters needed to be added unconditionally, which is checked by @RafkoNetBuilder::io_pre_requisites_set.
 */
class RAFKO_EXPORT RafkoNetBuilder{
public:
  class KernelParameters;

  RafkoNetBuilder(const rafko_mainframe::RafkoSettings& settings)
  :  m_settings(settings)
  { }

  /**
   * @brief      RafkoNetBuilder::input_size: sets the number of expected inputs for the RafkoNet object to be built
   *
   * @param[in]  size  The size
   *
   * @return     builder reference for chaining
   */
  constexpr RafkoNetBuilder& input_size(std::uint32_t size){
    m_argInputSize = size;
    m_isInputSizeSet = true;
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
    m_argOutputNeuronNumber = size;
    m_isOutputNeuronNumberSet = true;
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
    m_argExpectedInputRange = std::abs(range);
    m_isExpectedInputRangeSet = true;
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
      m_argWeightIniter = initializer;
      m_isWeightInitializerSet = true;
    }else m_isWeightInitializerSet = false;
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
      m_argNeuronArray = arr;
      m_isNeuronArraySet = true;
    }else m_isNeuronArraySet = false;
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
      m_argWeightTable = table;
      m_isWeightTableSet = true;
    }else m_isWeightTableSet = false;
    return *this;
  }

  /**
   * @brief      Sets an optional argument which restricts transfer functions by layer ( usable with @create_layers )
   *
   * @param[in]  allowed_transfer_functions_by_layer  The allowed transfer functions by layer
   *
   * @return     builder reference for chaining
   */
  RafkoNetBuilder& allowed_transfer_functions_by_layer(std::vector<std::set<Transfer_functions> > filter){
    m_argAllowedTransferFunctionsByLayer = filter;
    m_isAllowedTransferFunctionsByLayerSet = true;
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
    m_argNeuronIndexRecurrence.push_back({layer_index, layer_neuron_index, past});
    return *this;
  }

  KernelParameters& layer_input_convolution(std::uint32_t layer_index){
    return std::get<1>( *std::get<0>(m_layerKernelInputParameters.insert({layer_index, KernelParameters(*this)})) );
  }

  /**
   * @brief      creates a Fully connected feedforward neural network based on the IO arguments and
   *             and function arguments. The structure is according to the provided layer sizes
   *             argument, where the neurons of a layer is after in the previous layers, and before
   *             the succeeding layer Neurons.
   *
   * @param[in]  arena_ptr                  The pointer to the arena to allocate the resulting network in.
   *                                        In case of nullptr the result shall be on the heap and ownership is of the caller of the function
   * @param[in]  layerSizes                 how many layers will there be in the result and how big are those layers going to be
   * @param[in]  transfer_function_filter   The allowed transfer functions per layer
   *
   * @return   the built neural network
   */
  RafkoNet* create_layers(google::protobuf::Arena* arena_ptr, std::vector<std::uint32_t> layer_sizes, std::vector<std::set<Transfer_functions>> transfer_function_filter = {});

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
  RafkoNet* create_layers(std::vector<std::uint32_t> layer_sizes, std::vector<std::set<Transfer_functions>> transfer_function_filter = {}){
    return create_layers(m_settings.get_arena_ptr(), layer_sizes, transfer_function_filter);
  }

  /**
   * @brief     creates a Fully connected feedforward neural network based on the IO arguments and
   *             and function arguments. The structure is according to the provided layer sizes
   *             argument, where the neurons of a layer is after in the previous layers, and before
   *             the succeeding layer Neurons.
   *
   * @param      previous                   A pointer to the @RafkoNet object to swap the newly generated network with.
   *                                        The provided pointer is advised to be pointing to an object allocated on a protobuf Arena
   * @param[in]  layerSizes                 how many layers will there be in the result and how big are those layers going to be
   * @param[in]  transfer_function_filter   The allowed transfer functions per layer
   *
   * @return   the built neural network
   */
  void build_create_layers_and_swap(
    RafkoNet* previous, std::vector<std::uint32_t> layer_sizes,
    std::vector<std::set<Transfer_functions>> transfer_function_filter = {}
  );

private:
  const rafko_mainframe::RafkoSettings& m_settings;

  /**
   * @brief   Helper variables to see if different required arguments are set inside the builder
   */
  bool m_isInputSizeSet = false;
  bool m_isOutputNeuronNumberSet = false;
  bool m_isExpectedInputRangeSet = false;
  bool m_isWeightTableSet = false;
  bool m_isWeightInitializerSet = false;
  bool m_isNeuronArraySet = false;
  bool m_isAllowedTransferFunctionsByLayerSet = false;

  /**
   * @brief   Helper variables for features and optional Neuron parameters
   */
  std::vector< std::set<Transfer_functions> > m_argAllowedTransferFunctionsByLayer;
  std::unordered_map<std::uint32_t, std::set<Neuron_group_features>> m_layerFeatures;
  std::unordered_map<std::uint32_t, KernelParameters> m_layerKernelInputParameters;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Input_functions> > m_argNeuronIndexInputFunctions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Transfer_functions> > m_argNeuronIndexTransferFunctions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,Spike_functions> > m_argNeuronIndexSpikeFunctions;
  std::vector< std::tuple<std::uint32_t,std::uint32_t,std::uint32_t> > m_argNeuronIndexRecurrence;

  /**
   * The absolute value of the amplitude of one average input datapoint. It supports weight initialization.
   */
  double m_argExpectedInputRange = TransferFunction::get_average_output_range(transfer_function_identity);

  /**
   * The array containing the neurons while RafkoNetBuilder::build is used
   */
  std::vector<Neuron> m_argNeuronArray;

  /**
   * The array containing the used weights in the network while RafkoNetBuilder::build is used
   */
  std::vector<double> m_argWeightTable;

  /**
   * Weight Initializer argument, which guides the initial net Weights
   */
  std::shared_ptr<WeightInitializer> m_argWeightIniter;

  /**
   * Number of inputs the network to be built shall accept
   */
  std::uint32_t m_argInputSize = 0;

  /**
   * Number of Neurons the network to be built shall have as output
   */
  std::uint32_t m_argOutputNeuronNumber = 0;

public:
  class KernelParameters{
  public:
    KernelParameters(RafkoNetBuilder& parent)
    : m_parent(parent)
    , m_input({})
    , m_kernel({})
    , m_output({})
    { }

    /**
     * @brief    sets the dimension of the kernel to base the convolution on
     *
     * @param[in]...    sizes     The dimensions to set
     *
     * @return    reference to the object for chaining
     */
    template <typename ...Args>
    KernelParameters& kernel_size(Args... sizes){
      if(!checkDimensionCount(sizeof...(Args)))
        throw std::runtime_error("Wrong Dimensionality for Kernel Argument in kernelSize!");
      new (&m_kernel) rafko_utilities::NDArrayIndex({static_cast<std::uint32_t>(sizes) ...});
      return *this;
    }

    /**
     * @brief    sets the steps in input to step for one step in the output
     *
     * @param[in]...    steps     The number of steps to set
     *
     * @return    reference to the object for chaining
     */
    template <typename ...Args>
    KernelParameters& kernel_stride(Args... steps){
      if(!checkDimensionCount(sizeof...(Args)))
        throw std::runtime_error("Wrong Dimensionality for Kernel Argument in kernelStride!");
      m_kernelStride = {static_cast<std::uint32_t>(steps) ...};
      return *this;
    }

    /**
     * @brief    sets the padding values to base the convolution on
     *
     * @param[in]...    sizes     The padding sizes to set
     *
     * @return    reference to the object for chaining
     */
    template <typename ...Args>
    KernelParameters& input_padding(Args... sizes){
      if(!checkDimensionCount(sizeof...(Args)))
        throw std::runtime_error("Wrong Dimensionality for Kernel Argument in kernelPadding!");
      m_inputPadding = {static_cast<std::int32_t>(sizes) ...};
      if(0 < m_input.size())
        new (&m_input) rafko_utilities::NDArrayIndex(std::move(m_input), m_inputPadding);
      return *this;
    }

    /**
     * @brief    sets the dimension of the kernel input to base the convolution on
     *
     * @param[in]...    sizes     The dimensions to set
     *
     * @return    reference to the object for chaining
     */
    template <typename ...Args>
    KernelParameters& input_size(Args... sizes){
      if(!checkDimensionCount(sizeof...(Args)))
        throw std::runtime_error("Wrong Dimensionality for Kernel Argument in inputSize!");
      if(m_inputPadding.size() == sizeof...(Args))
        new (&m_input) rafko_utilities::NDArrayIndex({static_cast<std::uint32_t>(sizes) ...}, m_inputPadding);
        else new (&m_input) rafko_utilities::NDArrayIndex({static_cast<std::uint32_t>(sizes) ...});
      return *this;
    }
    
    /**
     * @brief    sets the dimension of the kernel output to base the convolution on
     *
     * @param[in]...    sizes     The dimensions to set
     *
     * @return    reference to the object for chaining
     */
    template <typename ...Args>
    KernelParameters& output_size(Args... sizes){
      if(!checkDimensionCount(sizeof...(Args)))
        throw std::runtime_error("Wrong Dimensionality for Kernel Argument in outputSize!");
      new (&m_output) rafko_utilities::NDArrayIndex({static_cast<std::uint32_t>(sizes) ...});
      return *this;
    }

    /**
     * @brief    validates the input, output and kernels with the strides and optional padding,
     *           calculates either the input or output dimensions if one of them is not available
     *
     * @return reference to the parent object for chaining
     */
    RafkoNetBuilder& validate();

    /**
     * @brief     Resets all stored parameters
     *
     * @return    reference to the object for chaining
     */
    KernelParameters& reset(){
      m_dimensionCount = 0;
      new (&m_input) rafko_utilities::NDArrayIndex({});
      new (&m_kernel) rafko_utilities::NDArrayIndex({});
      new (&m_output) rafko_utilities::NDArrayIndex({});
      m_inputPadding.clear();
      m_kernelStride.clear();
      m_valid = false;
      return *this;
    }

    /**
     * @brief     Provides the stored input parameter
     *
     * @return    reference of the stored input parameter as a non-const reference, so iteration is possible by it
     */
    rafko_utilities::NDArrayIndex& input(){
      if(!m_valid)throw std::runtime_error("Asked for input dimensions in invalid state!");
      return m_input;
    }

    /**
     * @brief     Provides the stored strides parameter; set for each dimension of the input parameter
     *
     * @return    reference of the stored strides parameter
     */
    const std::vector<std::uint32_t>& stride(){
      if(!m_valid)throw std::runtime_error("Asked for kernel stride in invalid state!");
      return m_kernelStride;
    }

    /**
     * @brief     Provides the stored kernel parameter
     *
     * @return    reference of the stored kernel parameter as a non-const reference, so iteration is possible by it
     */
    rafko_utilities::NDArrayIndex& kernel(){
      if(!m_valid)throw std::runtime_error("Asked for kernel dimensions in invalid state!");
      return m_kernel;
    }

    /**
     * @brief     Provides the stored output parameter
     *
     * @return    reference of the stored output parameter as a non-const reference, so iteration is possible by it
     */
    rafko_utilities::NDArrayIndex& output(){
      if(!m_valid)throw std::runtime_error("Asked for output dimensions in invalid state!");
      return m_output;
    }

  private:
    RafkoNetBuilder& m_parent;
    std::uint32_t m_dimensionCount = 0u;
    std::vector<std::int32_t> m_inputPadding;
    std::vector<std::uint32_t> m_kernelStride;
    bool m_valid = false;

    rafko_utilities::NDArrayIndex m_input;
    rafko_utilities::NDArrayIndex m_kernel;
    rafko_utilities::NDArrayIndex m_output;

    /**
     * @brief     Utility function to check if the provided dimension count matches the stored parameter if the stored
     *            parameter is above 0. If the stored dimensions count is zero, it is being updated with the provided value
     *
     * @return    True if the given dimension count is compatible with the stored one
     **/
    bool checkDimensionCount(std::uint32_t dim){
      if(0 == m_dimensionCount)m_dimensionCount = dim;
      return dim == m_dimensionCount;
    }

    /**
     * @brief     Checks if all parameters are set to be able to validate the object
     *
     * @return    true, if all required parameters are set to validate the object
     */
    bool check_kernel_complete(){
      return ( (0 < m_kernel.size())&&(0 < m_kernelStride.size())&&((0 < m_input.size())||(0 < m_output.size())) );
    }

    /**
     * @brief     Compares the stored object parameters to decide if the set convolution is computable
     *
     * @return    True, if the input size matches the size calculated by iterating on the output by the kernel
     */
    bool check_kernel_sizes();
  }; /* class RafkoNetBuilder::KernelParameters */
}; /* class RafkoNetBuilder */

} /* namespace rafko_net */
#endif /* RAFKO_NET_BUILDER_H */
