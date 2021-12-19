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
#include "rafko_net/models/rafko_softmax_feature.h"

#include <limits>

namespace rafko_net {

sdouble32 RafkoSoftmaxFeature::get_maximum_from(const std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads){
  sdouble32 max_value = -std::numeric_limits<double>::max();
  execution_threads.start_and_block([&max_value, neuron_data, relevant_neurons](uint32 thread_index){
    /* find local maximum */
    /* see if current maximum is greater, than global, and if so, ovewrite */
  });
  return max_value;
}

void RafkoSoftmaxFeature::calculate(std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads){
  sdouble32 max_value = get_maximum_from(neuron_data, relevant_neurons, execution_threads);
  /* modify the value of each Neuron */
}

} /* namespace rafko_net */
