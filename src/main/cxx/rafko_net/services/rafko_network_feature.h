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

#include "rafko_global.h"

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
#include "rafko_utilities/services/thread_group.h"
#include "rafko_mainframe/models/rafko_settings.h"

namespace rafko_net{

/**
 * @brief      A base class for all RafkoNeuralNetwork related features
 */
class RafkoNetworkFeature{
public:
  RafkoNetworkFeature(std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads_)
  : execution_threads(execution_threads_)
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
    std::vector<sdouble32>& neuron_data, uint32 thread_index = 0
  ) const;

  /**
   * @brief     Execute the given @FeatureGroup(supposedly performance relevant) on the provided neuron data buffer
   *
   * @param[in]  feature        The feature containing the is and the relevant neurons
   * @param[in]  settings       The settings object containing the required hyperparameters for the features
   * @param[in]  network        The network to calculate the values from
   * @param[in]  thread_index   The index of the thread the feature is to be executed
   */
  sdouble32 calculate_performance_relevant(
    const FeatureGroup& feature, const rafko_mainframe::RafkoSettings& settings,
    const RafkoNet& network, uint32 thread_index = 0
  ) const;


  #if(RAFKO_USES_OPENCL)
  /**
   * @brief      Provide the calculations of the given feature group as GPU kernel code
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  settings             The settings object containing the required hyperparameters for the features
   * @param[in]  solution             The feature group relevant solution
   * @param[in]  input_start_index    Variable to help with indexing
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_kernel_code_to(
    std::string& operations, const FeatureGroup& feature,
    const rafko_mainframe::RafkoSettings& settings, const Solution& solution,
    std::string input_start_index = "", std::string output_start_index = "",
    bool declare_locals = true
  );
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads;
  #if(RAFKO_USES_OPENCL)
  static std::mutex feature_cache_mutex;
  static uint32 l1_feature_called;
  static uint32 l2_feature_called;
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
    std::function<void(uint32)>&& fun, uint32 thread_index = 0u
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
    std::vector<sdouble32>& neuron_data,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    uint32 thread_index = 0u
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
    std::vector<sdouble32>& neuron_data, const rafko_mainframe::RafkoSettings& settings,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    uint32 thread_index = 0u
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
  sdouble32 calculate_l1_regularization(
    const rafko_net::RafkoNet& network,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    uint32 thread_index = 0u
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
  sdouble32 calculate_l2_regularization(
    const rafko_net::RafkoNet& network,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
    uint32 thread_index = 0u
  ) const;

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief      Provide the softmax function GPU Kernel.
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_softmax_kernel_to(
    std::string& operations, const FeatureGroup& feature,
    std::string output_start_index = "", bool declare_locals = true
  );

  /**
   * @brief      Provide the dropout function GPU Kernel.
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  settings             The settings object containing the required hyperparameters for the features
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  output_start_index   Variable to help with indexing
   */
  static void add_dropout_kernel_to(
    std::string& operations, const rafko_mainframe::RafkoSettings& settings,
    const FeatureGroup& feature, std::string output_start_index = ""
  );

  /**
   * @brief      Provide the l1 weight regularization GPU Kernel parts into the provided operations argument.
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  solution             The feature group relevant solution
   * @param[in]  input_start_index    Variable to help with indexing
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_l1_kernel_to(
    std::string& operations, const FeatureGroup& feature, const Solution& solution,
    std::string input_start_index = "", std::string output_start_index = "",
    bool declare_locals = true
  );

  /**
   * @brief      Provide the l2 weight regularization GPU Kernel parts into the provided operations argument.
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  solution             The feature group relevant solution
   * @param[in]  input_start_index    Variable to help with indexing
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_l2_kernel_to(
    std::string& operations, const FeatureGroup& feature, const Solution& solution,
    std::string input_start_index = "", std::string output_start_index = "",
    bool declare_locals = true
  );

  /**
   * @brief      The common part of L1 and L2 weight regularization
   *
   * @param      operations           The string to append the relevant operations into
   * @param[in]  lx                   The operation to use on each weight
   * @param[in]  local_name           Name of the local variable to be used in the feature
   * @param[in]  feature              The Neuron group feature to generate the kernel code for
   * @param[in]  solution             The feature group relevant solution
   * @param[in]  input_start_index    Variable to help with indexing
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_lx_kernel_to(
    std::string& operations, std::function<std::string(std::string)>&& lx, std::string local_name,
    const FeatureGroup& feature, const Solution& solution,
    std::string input_start_index = "", std::string output_start_index = "",
    bool declare_locals = true
  );
  #endif/*(RAFKO_USES_OPENCL)*/

};

} /* namespace rafko_net */

#endif /* RAFKO_NETWORK_FEATURE_H */
