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

#ifndef RAFKO_NETWORK_FEATURE_H
#define RAFKO_NETWORK_FEATURE_H

#include "rafko_global.hpp"

#include <vector>
#include <memory>
#include <functional>
#if(RAFKO_USES_OPENCL)
#include <string>
#include <mutex>
#include <regex>

#include "rafko_protocol/solution.pb.h"
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/models/subscript_proxy.hpp"
#include "rafko_utilities/services/thread_group.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"

namespace rafko_net{

/**
 * @brief      A base class for all RafkoNeuralNetwork related features
 */
class RAFKO_EXPORT RafkoNetworkFeature{
public:
  using NeuronDataProxy = rafko_utilities::SubscriptProxy<std::vector<double>>;

  RafkoNetworkFeature(std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads)
  : m_executionThreads(execution_threads)
  { }

  /**
   * @brief     Execute the given @FeatureGroup(supposedly solution relevant) on the provided neuron data buffer
   *
   * @param[in]  feature        The feature containing the is and the relevant neurons
   * @param[in]  settings       The settings object containing the required hyperparameters for the features
   * @param      neuron_data    The array containing the neuron data to update
   * @param[in]  thread_index   The index of the thread the feature is to be executed
   */
  void execute_solution_relevant(
    const FeatureGroup& feature, const rafko_mainframe::RafkoSettings& settings,
    NeuronDataProxy neuron_data, std::uint32_t thread_index = 0
  ) const;

  /**
   * @brief     Execute the given @FeatureGroup(supposedly performance relevant) on the provided neuron data buffer
   *
   * @param[in]  feature        The feature containing the is and the relevant neurons
   * @param[in]  settings       The settings object containing the required hyperparameters for the features
   * @param[in]  network        The network to calculate the values from
   * @param[in]  thread_index   The index of the thread the feature is to be executed
   */
  double calculate_performance_relevant(
    const FeatureGroup& feature, const rafko_mainframe::RafkoSettings& settings,
    const RafkoNet& network, std::uint32_t thread_index = 0
  ) const;

  #if(RAFKO_USES_OPENCL)

  /**
   * @brief      Provide the calculations of the given feature group as GPU kernel code.
   *             Called 'default' because in terms of indexing neuron index means the index of its data.
   *             The relevant index values are generated based on the index values of the provided @feature_group argument
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  feature_group        The Neuron group feature to generate the kernel code for
   * @param[in]  settings             The settings object containing the required hyperparameters for the features
   * @param[in]  solution             The feature group relevant solution
   * @param[in]  input_array          Might mean the weights array or the input array, etc..
   * @param[in]  input_start_index    Index helper: Might mean the start of the weights array or the input array, etc..
   * @param[in]  output_array         Might mean the neuron array, weight derivatives etc..
   * @param[in]  output_start_index   Index helper: Might mean the start of the neuron array, weight derivatives etc..
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_default_kernel_code_to(
    std::string& operations, const FeatureGroup& feature_group,
    const rafko_mainframe::RafkoSettings& settings, const Solution& solution,
    std::string input_array, std::string input_start_index,
    std::string output_array, std::string output_start_index,
    bool declare_locals
  );

  /**
   * @brief      Provide the calculations of the given feature group as GPU kernel code
   *
   * @param[in]  settings                 The settings object containing the required hyperparameters for the features
   * @param[in]  feature                  The feature to generate the kernel code for
   * @param[in]  relevant_index_values    The index values the feature is relevant to
   * @param[in]  input_array              Might mean the weights array or the input array, etc..
   * @param[in]  input_start_index        Index helper: Might mean the start of the weights array or the input array, etc..
   * @param[in]  output_array             Might mean the neuron array, weight derivatives etc..
   * @param[in]  output_start_index       Index helper: Might mean the start of the neuron array, weight derivatives etc..
   * @param[in]  declare_locals           Decides whether local variables used by the kernel are to be declared or just updated
   */
  static std::string generate_kernel_code(
    const rafko_mainframe::RafkoSettings& settings, Neuron_group_features feature,
    const std::vector<std::uint32_t>& relevant_index_values,
    std::string input_array, std::string input_start_index,
    std::string output_array, std::string output_start_index,
    bool declare_locals = true
  );

  static std::string get_kernel_locals(){
    return R"(
      double exp_sum = 0.0;
      double l1_error = 0.0;
      double l2_error = 0.0;
      uint dropout_seed = 0;
    )";
  }

  #endif/*(RAFKO_USES_OPENCL)*/

private:
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& m_executionThreads;
  #if(RAFKO_USES_OPENCL)
  static inline std::mutex m_featureCacheMutex;
  static inline std::uint32_t m_lxFeatureCalled;
  #endif/*(RAFKO_USES_OPENCL)*/

  /**
   * @brief     Execute the provided function for every relevant Neuron in a multi-threaded environment
   *
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  fun                The function to call with the index value of every relevant Neuron
   * @param[in]  thread_index       The index of the thread the feature is to be executed
   */
  void execute_in_paralell_for(
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    std::function<void(std::uint32_t)>&& fun, std::uint32_t thread_index = 0u
  ) const;

  /**
   * @brief      Calculate the softmax function by setting the data values provided in the arguments
   *             to be of sum of one.
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  thread_index       The index of the thread the feature is to be executed
   */
  void execute_softmax(
    NeuronDataProxy neuron_data,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    std::uint32_t thread_index = 0u
  ) const;

  /**
   * @brief      Calculate the dropout function for the given group of Neurons which randomly sets neuron activation values to zero.
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  settings           The settings object containing the required hyperparameters for the features
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  thread_index       The index of the thread the feature is to be executed
   */
  void execute_dropout(
    NeuronDataProxy neuron_data,
    const rafko_mainframe::RafkoSettings& settings,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    std::uint32_t thread_index = 0u
  ) const;


  /**
   * @brief      Calculate the error value coming from L1 weight regularization
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  thread_index       The index of the thread the feature is to be executed
   *
   * @return the resulting error value
   */
  double calculate_l1_regularization(
    const rafko_net::RafkoNet& network,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    std::uint32_t thread_index = 0u
  ) const;

  /**
   * @brief      Calculate the error value coming from L2 weight regularization
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  thread_index       The index of the thread the feature is to be executed
   *
   * @return the resulting error value
   */
  double calculate_l2_regularization(
    const rafko_net::RafkoNet& network,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    std::uint32_t thread_index = 0u
  ) const;

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief      Provide the softmax function GPU Kernel.
   *
   * @param      operations                 The string to append the relevant operations into
   * @param[in]  relevant_index_values      The relevant index values of the given arrays to base kernel operations on
   * @param[in]  neuron_data_array          The Neuron array containing the data values to softmax
   * @param[in]  neuron_data_start_index    The start of the neuron array
   * @param[in]  declare_locals             Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_softmax_kernel_to(
    std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
    std::string neuron_data_array, std::string neuron_data_start_index, bool declare_locals
  );

  /**
   * @brief      Provide the dropout function GPU Kernel.
   *
   * @param      operations                 The string to append the relevant operations into
   * @param[in]  settings                   The settings object containing the required hyperparameters for the features
   * @param[in]  relevant_index_values      The relevant index values of the given arrays to base kernel operations on
   * @param[in]  neuron_data_array          The Neuron array containing the data values to softmax
   * @param[in]  neuron_data_start_index    The start of the neuron array
   * @param[in]  declare_locals             Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_dropout_kernel_to(
    std::string& operations, const rafko_mainframe::RafkoSettings& settings,
    const std::vector<std::uint32_t>& relevant_index_values,
    std::string neuron_data_array, std::string neuron_data_start_index, bool declare_locals
  );

  /**
   * @brief      Provide the l1 weight regularization GPU Kernel parts into the provided operations argument.
   *
   * @param      operations                 The string to append the relevant operations into
   * @param[in]  relevant_index_values      The relevant index values of the given arrays to base kernel operations on
   * @param[in]  weight_array               The buffer containing the target to read weight info from
   * @param[in]  weight_start_index         Variable to help with indexing
   * @param[in]  output_array               The buffer containing the target to write the calcualted error into
   * @param[in]  output_start_index         The start of the output array
   * @param[in]  declare_locals             Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_l1_kernel_to(
    std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
    std::string weight_array, std::string weight_start_index,
    std::string output_array, std::string output_start_index, bool declare_locals
  );

  /**
   * @brief      Provide the l2 weight regularization GPU Kernel parts into the provided operations argument.
   *
   * @param      operations                 The string to append the relevant operations into
   * @param[in]  relevant_index_values      The relevant index values of the given arrays to base kernel operations on
   * @param[in]  weight_array               The buffer containing the target to read weight info from
   * @param[in]  weight_start_index         Variable to help with indexing
   * @param[in]  output_array               The buffer containing the target to write the calcualted error into
   * @param[in]  output_start_index         The start of the output array
   * @param[in]  declare_locals             Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_l2_kernel_to(
    std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
    std::string weight_array, std::string weight_start_index,
    std::string output_array, std::string output_start_index, bool declare_locals
  );

  /**
   * @brief      The common part of L1 and L2 weight regularization
   *
   * @param      operations                 The string to append the relevant operations into
   * @param[in]  lx                         The operation to use on each weight
   * @param[in]  local_name                 Name of the local variable to be used in the feature
   * @param[in]  relevant_index_values      The relevant index values of the given arrays to base kernel operations on
   * @param[in]  weight_start_index         Variable to help with indexing
   * @param[in]  output_start_index         Variable to help with indexing
   * @param[in]  declare_locals             Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_lx_kernel_to(
    std::string& operations, std::function<std::string(std::string)>&& lx, std::string local_name,
    const std::vector<std::uint32_t>& relevant_index_values,
    std::string weight_array, std::string weight_start_index,
    std::string output_array, std::string output_start_index,
    bool declare_locals
  );
  #endif/*(RAFKO_USES_OPENCL)*/

};

} /* namespace rafko_net */

#endif /* RAFKO_NETWORK_FEATURE_H */
