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

#include "rafko_net/services/rafko_network_feature.h"

#include <limits>
#include <math.h>
#if(RAFKO_USES_OPENCL)
#include <string>
#include <regex>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net {

void RafkoNetworkFeature::execute_for_relevant_neurons(
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons, std::function<void(uint32)> fun
) const{
  execution_threads[0]->start_and_block([this, &relevant_neurons, &fun](uint32 thread_index){
    SynapseIterator<> relevant_neuron_iterator(relevant_neurons);
    uint32 neurons_to_do_in_one_thread = 1u + (relevant_neuron_iterator.size()/execution_threads[0]->get_number_of_threads());
    uint32 start_index = std::min( relevant_neuron_iterator.size(), (neurons_to_do_in_one_thread * thread_index) );
    uint32 neurons_to_do_in_this_thread = std::min(neurons_to_do_in_one_thread, (relevant_neuron_iterator.size() - start_index));
    for(uint32 synapse_index = 0; synapse_index < neurons_to_do_in_this_thread; synapse_index++){
      fun(relevant_neuron_iterator[start_index + synapse_index]);
    }
  });
}

void RafkoNetworkFeature::execute(const FeatureGroup& feature, std::vector<sdouble32>& neuron_data, uint32 thread_index) const{
  assert(thread_index < execution_threads.size());

  switch(feature.feature()){
    case neuron_group_feature_softmax: calculate_softmax(neuron_data, feature.relevant_neurons());
      break;
    default: break;
  }
}

void RafkoNetworkFeature::calculate_softmax(
  std::vector<sdouble32>& neuron_data,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons
) const{
  std::atomic<sdouble32> max_value{-std::numeric_limits<double>::max()};
  std::atomic<sdouble32> expsum{double_literal(0)};

  execute_for_relevant_neurons(relevant_neurons, [&max_value, &expsum, &neuron_data](uint32 neuron_index){
    sdouble32 current = max_value.load();
    while(
      (neuron_data[neuron_index] > current)
      &&(!max_value.compare_exchange_weak(current,neuron_data[neuron_index]))
    )current = max_value;

    current = expsum;
    while(
      !expsum.compare_exchange_weak(current, (current + std::exp(neuron_data[neuron_index])))
    ) current = expsum;
  });

  sdouble32 used_max_value = max_value;
  sdouble32 used_expsum = expsum / std::exp(max_value);
  /*!Note: to make the softmax function numerically stable, the maximum value is subtracted from all values, and the sum is corrected for that.
   * Because the maximum value is not known during the calculation of the sum, it needs to be corrected by dividing the sum with exp(max(x)).
   * This is possible, because expq(x-c) = exp(x) / exp(c) for every element in the sum.
   */
  used_expsum = std::max(used_expsum, std::numeric_limits<double>::epsilon());
  /*!Note: Since no Neuron will be involved in the execution twice, no need for a mutual exclusive lock or atomics */
  execute_for_relevant_neurons(relevant_neurons,[&neuron_data, &used_max_value, &used_expsum](uint32 neuron_index){
    neuron_data[neuron_index] = std::exp(neuron_data[neuron_index] - used_max_value) / used_expsum;
  });
}


#if(RAFKO_USES_OPENCL)
void RafkoNetworkFeature::add_kernel_code_to(
  std::string& operations, const FeatureGroup& feature,
  std::string output_start_index, bool declare_locals
){
  switch(feature.feature()){
    case neuron_group_feature_softmax:
      add_softmax_code_to(operations, feature, output_start_index, declare_locals);
      break;
    default: break;
  }
}

void RafkoNetworkFeature::add_softmax_code_to(std::string& operations, const FeatureGroup& feature, std::string output_start_index, bool declare_locals){
  SynapseIterator<> synapse_iterator(feature.relevant_neurons());
  std::regex index_regex("\\$\\$index\\$\\$");

  std::function<std::string(uint32)> index_string = [output_start_index](uint32 index){
    return (output_start_index + " + " + std::to_string(index));
  }; /*!Note: merges the provided output start index and the neuron index and provides an addition */

  std::function<std::string(std::string,uint32)> process_index_values = [output_start_index, index_string, index_regex](std::string input_text, uint32 index){
    uint32 matches_count = 1u; /* to start the iteration */
    std::string text = input_text;
    while(0u < matches_count){
      text = std::regex_replace(text, index_regex, index_string(index));
      matches_count = std::distance( /* https://stackoverflow.com/questions/8283735/count-number-of-matches */
        std::sregex_iterator(text.begin(), text.end(), index_regex),
        std::sregex_iterator()
      );
    }
    return text;
  }; /*!Note: replaces index regex until there are no more matches */

  /* Add the declaration into the  kernel */
  if(declare_locals){
    operations += R"( double max_val = DBL_MIN;
      double exp_sum = 0.0;
    )";
  }else{
    operations += R"( max_val = DBL_MIN;
      exp_sum = 0.0;
    )";
  }

  /* find max and expsum */
  synapse_iterator.iterate([&operations, process_index_values](uint32 neuron_index){
    operations += process_index_values(R"(max_val = max(max_val, outputs[$$index$$]);
       exp_sum += exp(outputs[$$index$$]);
     )", neuron_index);
  });

  /* epsilon guard */
  operations += "exp_sum = max(exp_sum, DBL_EPSILON);\n";

  /* apply softmax */
  synapse_iterator.iterate([&operations, process_index_values](uint32 neuron_index){
    operations += process_index_values("outputs[$$index$$] = exp(outputs[$$index$$] - max_val) / (exp_sum / exp(max_val));\n\n", neuron_index);
  });

  operations += "\n\n";
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
