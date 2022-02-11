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

#ifndef RAFKO_SOFTMAX_FEATURE_H
#define RAFKO_SOFTMAX_FEATURE_H

#include "rafko_global.h"

#include <vector>
#include <utility>
#include <google/protobuf/repeated_field.h>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/services/thread_group.h"

namespace rafko_net{

/**
 * @brief      Collector for softmax related functions and utilities
 */
class RafkoSoftmaxFeature{
public:

  /**
   * @brief      Calculate the softmax function by setting the data values provided in the arguments
   *             to be of sum of one.
   *
   * @param      neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  execution_threads  Used execution threads
   */
  static void calculate(std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads);

  #if(RAFKO_USES_OPENCL)
  static void add_kernel_code_to( std::string& operations, const FeatureGroup& feature, std::string output_start_index, bool declare_locals = true );
  #endif/*(RAFKO_USES_OPENCL)*/

private:

  /**
   * @brief      Calculate the maximum value and the sum of exp(x) for the relevant Neurons
   *
   * @param[in]  neuron_data        The array containing the neuron data to update
   * @param[in]  relevant_neurons   The index values of the relevant neurons to apply the function on
   * @param[in]  execution_threads  Used execution threads
   *
   * @return     an std::pair of {maximum value of the Neuron data values, the sum of e^(Neuron values))};
   */
  static std::pair<sdouble32,sdouble32> get_max_and_expsum(const std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads);

};

} /* namespace rafko_net */

#endif /* RAFKO_SOFTMAX_FEATURE_H */
