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
#include <unordered_map>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_net/services/synapse_iterator.h"
#if(RAFKO_USES_OPENCL)
#include "rafko_utilities/services/rafko_string_utils.h"
#include "rafko_gym/services/rafko_weight_updater.h"
#endif/*(RAFKO_USES_OPENCL)*/


namespace rafko_net {

void RafkoNetworkFeature::execute_in_paralell_for(
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::function<void(uint32)>&& fun, uint32 thread_index
) const{
  execution_threads[thread_index]->start_and_block([this, &relevant_neurons, &fun](uint32 thread_index){
    SynapseIterator<> relevant_neuron_iterator(relevant_neurons);
    uint32 neurons_to_do_in_one_thread = 1u + (relevant_neuron_iterator.size()/execution_threads[0]->get_number_of_threads());
    uint32 start_index = std::min( relevant_neuron_iterator.size(), (neurons_to_do_in_one_thread * thread_index) );
    uint32 neurons_to_do_in_this_thread = std::min(neurons_to_do_in_one_thread, (relevant_neuron_iterator.size() - start_index));
    for(uint32 synapse_index = 0; synapse_index < neurons_to_do_in_this_thread; synapse_index++){
      fun(relevant_neuron_iterator[start_index + synapse_index]);
    }
  });
}

void RafkoNetworkFeature::execute_solution_relevant(
  const FeatureGroup& feature, std::vector<sdouble32>& neuron_data, uint32 thread_index
) const{
  assert(thread_index < execution_threads.size());

  switch(feature.feature()){
    case neuron_group_feature_softmax: calculate_softmax(neuron_data, feature.relevant_neurons(), thread_index);
      break;
    default: break;
  }
}

sdouble32 RafkoNetworkFeature::calculate_performance_relevant(
  const FeatureGroup& feature, const RafkoNet& network, uint32 thread_index
) const{
  assert(thread_index < execution_threads.size());

  switch(feature.feature()){
    case neuron_group_feature_l1_regularization: return calculate_l1_regularization(network, feature.relevant_neurons(), thread_index);
    case neuron_group_feature_l2_regularization: return calculate_l2_regularization(network, feature.relevant_neurons(), thread_index);
    default: return double_literal(0.0);
  }
}


void RafkoNetworkFeature::calculate_softmax(
  std::vector<sdouble32>& neuron_data,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  uint32 thread_index
) const{
  std::atomic<sdouble32> max_value{-std::numeric_limits<double>::max()};
  std::atomic<sdouble32> expsum{double_literal(0)};

  execute_in_paralell_for(relevant_neurons, [&max_value, &expsum, &neuron_data](uint32 neuron_index){
    sdouble32 current = max_value.load();
    while(
      (neuron_data[neuron_index] > current)
      &&(!max_value.compare_exchange_weak(current,neuron_data[neuron_index]))
    )current = max_value;

    current = expsum;
    while(!expsum.compare_exchange_weak(current, (current + std::exp(neuron_data[neuron_index]))))
      current = expsum;
  }, thread_index);

  sdouble32 used_max_value = max_value;
  sdouble32 used_expsum = expsum / std::exp(max_value);
  /*!Note: to make the softmax function numerically stable, the maximum value is subtracted from all values, and the sum is corrected for that.
   * Because the maximum value is not known during the calculation of the sum, it needs to be corrected by dividing the sum with exp(max(x)).
   * This is possible, because expq(x-c) = exp(x) / exp(c) for every element in the sum.
   */
  used_expsum = std::max(used_expsum, std::numeric_limits<double>::epsilon());
  /*!Note: Since no Neuron will be involved in the execution twice, no need for a mutual exclusive lock or atomics */
  execute_in_paralell_for(relevant_neurons,[&neuron_data, &used_max_value, &used_expsum](uint32 neuron_index){
    neuron_data[neuron_index] = std::exp(neuron_data[neuron_index] - used_max_value) / used_expsum;
  }, thread_index);
}

sdouble32 RafkoNetworkFeature::calculate_l1_regularization(
  const rafko_net::RafkoNet& network,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  uint32 thread_index
) const{
  std::atomic<sdouble32> error_value = double_literal(0.0);
  execute_in_paralell_for(relevant_neurons,[&error_value, &network](uint32 neuron_index){
    SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
    [&error_value, &network](uint32 weight_index){
      sdouble32 current = error_value;
      while(!error_value.compare_exchange_weak(current, (error_value + std::abs(network.weight_table(weight_index)))))
        current = error_value;
    });
  }, thread_index);
  return static_cast<sdouble32>(error_value);
}

sdouble32 RafkoNetworkFeature::calculate_l2_regularization(
  const rafko_net::RafkoNet& network,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  uint32 thread_index
) const{
  std::atomic<sdouble32> error_value = double_literal(0.0);
  execute_in_paralell_for(relevant_neurons,[&error_value, &network](uint32 neuron_index){
    SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
    [&error_value, &network](uint32 weight_index){
      sdouble32 current = error_value;
      while(!error_value.compare_exchange_weak(
        current, ( error_value + std::pow(network.weight_table(weight_index), double_literal(2.0)) )
      ))current = error_value;
    });
  }, thread_index);
  return static_cast<sdouble32>(error_value);
}

#if(RAFKO_USES_OPENCL)
void RafkoNetworkFeature::add_kernel_code_to(
  std::string& operations, const FeatureGroup& feature, const Solution& solution,
  std::string input_start_index, std::string output_start_index,
  bool declare_locals
){
  switch(feature.feature()){
    case neuron_group_feature_softmax:
      add_softmax_kernel_to(operations, feature, solution, input_start_index, output_start_index, declare_locals);
      break;
    case rafko_net::neuron_group_feature_l1_regularization:
      add_l1_kernel_to(operations, feature, solution, input_start_index, output_start_index, declare_locals);
      break;
    case rafko_net::neuron_group_feature_l2_regularization:
      add_l2_kernel_to(operations, feature, solution, input_start_index, output_start_index, declare_locals);
      break;
    default: break;
  }
}

void RafkoNetworkFeature::add_softmax_kernel_to(
  std::string& operations, const FeatureGroup& feature, const Solution& solution,
  std::string input_start_index, std::string output_start_index,
  bool declare_locals
){
  parameter_not_used(input_start_index);
  parameter_not_used(solution);
  SynapseIterator<> synapse_iterator(feature.relevant_neurons());
  std::regex index_regex("==index==");

  std::function<std::string(uint32)> index_string = [output_start_index](uint32 index){
    return (output_start_index + " + " + std::to_string(index));
  }; /*!Note: merges the provided output start index and the neuron index and provides an addition */

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
  synapse_iterator.iterate([&operations, &index_string, index_regex](uint32 neuron_index){
    operations += rafko_utilities::replace_all_in_string(R"(max_val = max(max_val, outputs[==index==]);
       exp_sum += exp(outputs[==index==]);
     )", index_regex, index_string(neuron_index));
  });

  /* epsilon guard */
  operations += "exp_sum = max(exp_sum, DBL_EPSILON);\n";

  /* apply softmax */
  synapse_iterator.iterate([&operations, &index_string, index_regex](uint32 neuron_index){
    operations += rafko_utilities::replace_all_in_string(
      "outputs[==index==] = exp(outputs[==index==] - max_val) / (exp_sum / exp(max_val));\n\n",
      index_regex, index_string(neuron_index)
    );
  });

  operations += "\n\n";
}

void RafkoNetworkFeature::add_l1_kernel_to(
  std::string& operations, const FeatureGroup& feature, const Solution& solution,
  std::string input_start_index, std::string output_start_index,
  bool declare_locals
){
  add_lx_kernel_to(operations, [](std::string input){
    return "fabs(" + input + ")";
  }, "l1_error",feature, solution, input_start_index, output_start_index, declare_locals);
}

void RafkoNetworkFeature::add_l2_kernel_to(
  std::string& operations, const FeatureGroup& feature, const Solution& solution,
  std::string input_start_index, std::string output_start_index,
  bool declare_locals
){
  add_lx_kernel_to(operations, [](std::string input){
    return "pow(" + input + ", 2.0)";
  }, "l2_error", feature, solution, input_start_index, output_start_index, declare_locals);
}

std::mutex RafkoNetworkFeature::feature_cache_mutex;
uint32 RafkoNetworkFeature::l1_feature_called = 0u;
void RafkoNetworkFeature::add_lx_kernel_to(
  std::string& operations, std::function<std::string(std::string)>&& lx, std::string local_name,
  const FeatureGroup& feature, const Solution& solution,
  std::string input_start_index, std::string output_start_index,
  bool declare_locals
){
  {
    std::lock_guard<std::mutex> my_lock(feature_cache_mutex);
    ++l1_feature_called;
    /*!Note: if the function were to be called after the start of the function, the index 0
     * would not be wasted, but this way the function is also thread-safe
     */
  }

  std::string feature_helpers;
  if(declare_locals){
    feature_helpers = R"(
      double ==local_name== = 0.0;
    )";
  }

  parameter_not_used(declare_locals);
  feature_helpers += R"(
    const int ==index_var_name==[==feature_weight_number==] = {
      ==index_values==
    };
  )";
  uint32 relevant_weight_count = 0u;
  std::string index_list;
  std::unordered_map<uint32, uint32> map_neurons_to_partials;
  std::unordered_map<uint32, uint32> map_weight_starts_to_partials;
  std::unordered_map<uint32, uint32> map_weight_sypase_start_to_neurons;
  SynapseIterator<>::iterate(feature.relevant_neurons(),
  [&](uint32 neuron_index){
    /* find neuron inside the partial solutions */
    uint32 partial_index = rafko_gym::RafkoWeightUpdater::get_relevant_partial_index_for(
      neuron_index, solution, map_neurons_to_partials
    );
    const PartialSolution& partial = solution.partial_solutions(partial_index);
    uint32 weight_start_in_partial = rafko_gym::RafkoWeightUpdater::get_device_weight_table_start_for(
      partial_index, solution, map_weight_starts_to_partials
    );
    uint32 inner_neuron_index = (neuron_index - solution.partial_solutions(partial_index).output_data().starts());
    uint32 weight_start_synapse = rafko_gym::RafkoWeightUpdater::get_weight_synapse_start_index_in_partial(
      neuron_index, partial, map_weight_sypase_start_to_neurons
    );
    SynapseIterator<>::iterate(partial.weight_indices(),
    [&relevant_weight_count](IndexSynapseInterval weight_interval){
      relevant_weight_count += weight_interval.interval_size();
    },
    [&index_list, input_start_index, weight_start_in_partial](uint32 weight_index){
      index_list += "(" + input_start_index + "+" + std::to_string(weight_start_in_partial + weight_index) + "),";
    }, weight_start_synapse, partial.weight_synapse_number(inner_neuron_index));
  });
  index_list.pop_back();
  std::string index_size_variable = "feature_lx_" + std::to_string(l1_feature_called) + "_index_values";
  std::string index_variable = "feature_lx_" + std::to_string(l1_feature_called) + "_index_values";

  /* add feature calculations */
  std::string feature_calculations = feature_helpers + R"(
    if(evaluate_network && (get_global_id(0) <= ==feature_weight_number==)){
      const int weights_in_thread = ( ==feature_weight_number==/(int)(get_global_size(0)) ) + 1;
      const int execution_start_index = weights_in_thread * get_global_id(0);

      ==local_name== = 0.0;
      for(
        int w_i = 0;
        (
          (w_i < weights_in_thread)
          &&((execution_start_index + w_i) < ==feature_weight_number==)
        );
        ++w_i
      ){
        ==local_name== += ==lx==;
      }

      AtomicAdd(&outputs[==output_start_index==], ==local_name==);
    }/*if(evaluating network)*/
  )";

  /* Replace helper variables */
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==lx=="),
    lx("inputs[==index_var_name==[execution_start_index + w_i]]")
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==index_values=="), index_list
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==index_var_name=="), index_variable
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==feature_weight_number=="),
    std::to_string(relevant_weight_count)
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==output_start_index=="),
    output_start_index
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==local_name=="),
    local_name
  );
  operations += feature_calculations;
}

#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
