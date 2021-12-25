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
#include <math.h>
#include <mutex>

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net {

std::pair<sdouble32,sdouble32> RafkoSoftmaxFeature::get_max_and_expsum(const std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads){
  std::mutex return_value_mutex;
  sdouble32 max_value = -std::numeric_limits<double>::max();
  sdouble32 expsum = double_literal(0); /*!Note: could use atomic, but there's already a mutex involved because of maximum */
  execution_threads.start_and_block([&execution_threads, &return_value_mutex, &max_value, &expsum, neuron_data, relevant_neurons](uint32 thread_index){
    sdouble32 thread_max_value = -std::numeric_limits<double>::max();
    sdouble32 thread_expsum = double_literal(0);
    SynapseIterator<> relevant_neuron_iterator(relevant_neurons);
    uint32 neurons_to_do_in_one_thread = 1u + (relevant_neuron_iterator.size()/execution_threads.get_number_of_threads());
    uint32 start_index = std::min( relevant_neuron_iterator.size(), (neurons_to_do_in_one_thread * thread_index) );
    uint32 neurons_to_do_in_this_thread = std::min(neurons_to_do_in_one_thread, (relevant_neuron_iterator.size() - start_index));
    for(uint32 synapse_index = 0; synapse_index < neurons_to_do_in_this_thread; synapse_index++){
      thread_expsum += std::exp(neuron_data[relevant_neuron_iterator[start_index + synapse_index]]);
      if(neuron_data[relevant_neuron_iterator[start_index + synapse_index]] > thread_max_value)
        thread_max_value = neuron_data[relevant_neuron_iterator[start_index + synapse_index]];
    }

    if(0u < neurons_to_do_in_this_thread){
      std::lock_guard<std::mutex> my_lock(return_value_mutex);
      expsum += thread_expsum;
      if(thread_max_value > max_value)
        max_value = thread_max_value;
    }
  });
  return std::make_pair(max_value, expsum);
}

void RafkoSoftmaxFeature::calculate(std::vector<sdouble32>& neuron_data, const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, rafko_utilities::ThreadGroup& execution_threads){
  std::pair<sdouble32,sdouble32> softmax_params = get_max_and_expsum(neuron_data, relevant_neurons, execution_threads);
  execution_threads.start_and_block([&execution_threads, &softmax_params, &neuron_data, &relevant_neurons](uint32 thread_index){
    SynapseIterator<> relevant_neuron_iterator(relevant_neurons);
    uint32 neurons_to_do_in_one_thread = 1u + (relevant_neuron_iterator.size()/execution_threads.get_number_of_threads());
    uint32 start_index = std::min( relevant_neuron_iterator.size(), (neurons_to_do_in_one_thread * thread_index) );
    uint32 neurons_to_do_in_this_thread = std::min(neurons_to_do_in_one_thread, (relevant_neuron_iterator.size() - start_index));

    for(uint32 synapse_index = 0; synapse_index < neurons_to_do_in_this_thread; synapse_index++){
      neuron_data[relevant_neuron_iterator[start_index + synapse_index]] = ( /*!Note: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python */
        std::exp(neuron_data[relevant_neuron_iterator[start_index + synapse_index]] - std::get<0>(softmax_params))
        / ( std::get<1>(softmax_params) / std::exp(std::get<0>(softmax_params)) )
      ); /* x = (x - max(x)) / (expsum(x) / exp(max(x))) */
      /*!Note: to make the softmax function numerically stable, the maximum value is subtracted from all values, and the sum is corrected for that.
       * Because the maximum value is not known during the calculation of the sum, it needs to be corrected by dividing the sum with exp(max(x)).
       * This is possible, because expq(x-c) = exp(x) / exp(c) for every element in the sum.
       */
    }
  });
}

} /* namespace rafko_net */
