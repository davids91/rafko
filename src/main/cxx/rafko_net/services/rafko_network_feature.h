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
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/services/thread_group.h"

namespace rafko_net{

/**
 * @brief      A base class for all RafkoNeuralNetwork related features
 */
class RafkoNetworkFeature{
public:
  RafkoNetworkFeature(std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads_)
  : execution_threads(execution_threads_)
  { }

  void execute_for_relevant_neurons(
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, std::function<void(uint32)> fun
  ) const;

  void execute(const FeatureGroup& feature, std::vector<sdouble32>& neuron_data, uint32 thread_index = 0) const;
  #if(RAFKO_USES_OPENCL)
  static void add_kernel_code_to(
    std::string& operations, const FeatureGroup& feature,
    std::string output_start_index, bool declare_locals = true
  );
  #endif/*(RAFKO_USES_OPENCL)*/

private:
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads;

  /**
   * @brief      Calculate the softmax function by setting the data values provided in the arguments
   *             to be of sum of one.
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  execution_threads  Used execution threads
   */
  void calculate_softmax(
    std::vector<sdouble32>& neuron_data,
    const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons
  ) const;

  #if(RAFKO_USES_OPENCL)
  /**
   * @brief      Provide the softmax function GPU Kernel.
   *
   * @param      operations           The string to append the relevant perations into
   * @param[in]  feature              The feature containing the is and the relevant neurons
   * @param[in]  output_start_index   Variable to help with indexing
   * @param[in]  declare_locals       Decides whether local variables used by the kernel are to be declared or just updated
   */
  static void add_softmax_code_to(
    std::string& operations, const FeatureGroup& feature,
    std::string output_start_index, bool declare_locals = true
  );
  #endif/*(RAFKO_USES_OPENCL)*/

};

} /* namespace rafko_net */

#endif /* RAFKO_NETWORK_FEATURE_H */
