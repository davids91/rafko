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
#include <google/protobuf/repeated_field.h>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_utilities/services/thread_group.h"

namespace rafko_net{

/**
 * @brief      Collector for softmax related functions and utilities
 */
class RafkoSoftmaxFeature{
public:
  static sdouble32 get_maximum_from(const std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads);
  static void calculate(std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads);
};

} /* namespace rafko_net */

#endif /* RAFKO_SOFTMAX_FEATURE_H */
