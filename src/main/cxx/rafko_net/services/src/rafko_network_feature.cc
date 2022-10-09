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

#include "rafko_net/services/rafko_network_feature.hpp"

#include <limits>
#include <math.h>
#if(RAFKO_USES_OPENCL)
#include <unordered_map>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#if(RAFKO_USES_OPENCL)
#include "rafko_utilities/services/rafko_string_utils.hpp"
#include "rafko_gym/services/rafko_weight_adapter.hpp"
#endif/*(RAFKO_USES_OPENCL)*/


namespace rafko_net {

void RafkoNetworkFeature::execute_in_paralell_for(
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::function<void(std::uint32_t)>&& fun, std::uint32_t thread_index
) const{
  m_executionThreads[thread_index]->start_and_block([this, &relevant_neurons, &fun](std::uint32_t thread_index){
    SynapseIterator<> relevant_neuron_iterator(relevant_neurons);
    std::uint32_t neurons_to_do_in_one_thread = 1u + (relevant_neuron_iterator.size()/m_executionThreads[0]->get_number_of_threads());
    std::uint32_t start_index = std::min( relevant_neuron_iterator.size(), (neurons_to_do_in_one_thread * thread_index) );
    std::uint32_t neurons_to_do_in_this_thread = std::min(neurons_to_do_in_one_thread, (relevant_neuron_iterator.size() - start_index));
    for(std::uint32_t synapse_index = 0; synapse_index < neurons_to_do_in_this_thread; synapse_index++){
      fun(relevant_neuron_iterator[start_index + synapse_index]);
    }
  });
}

void RafkoNetworkFeature::execute_solution_relevant(
  const FeatureGroup& feature, const rafko_mainframe::RafkoSettings& settings,
  NeuronDataProxy neuron_data, std::uint32_t thread_index
) const{
  RFASSERT(thread_index < m_executionThreads.size());

  switch(feature.feature()){
    case neuron_group_feature_softmax: execute_softmax(neuron_data, feature.relevant_neurons(), thread_index);
      break;
    case neuron_group_feature_dropout_regularization: execute_dropout(neuron_data, settings, feature.relevant_neurons(), thread_index);
      break;
    default: break;
  }
}

double RafkoNetworkFeature::calculate_performance_relevant(
  const FeatureGroup& feature, const rafko_mainframe::RafkoSettings& /*settings*/,
  const RafkoNet& network, std::uint32_t thread_index
) const{
  RFASSERT(thread_index < m_executionThreads.size());

  switch(feature.feature()){
    case neuron_group_feature_l1_regularization: return calculate_l1_regularization(network, feature.relevant_neurons(), thread_index);
    case neuron_group_feature_l2_regularization: return calculate_l2_regularization(network, feature.relevant_neurons(), thread_index);
    default: return 0.0;
  }
}


void RafkoNetworkFeature::execute_softmax(
  NeuronDataProxy neuron_data,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::uint32_t thread_index
) const{
  std::atomic<double> max_value{-std::numeric_limits<double>::max()};
  std::atomic<double> expsum{(0)};

  execute_in_paralell_for(relevant_neurons, [&max_value, &expsum, &neuron_data](std::uint32_t neuron_index){
    double current = max_value.load();
    while(
      (neuron_data[neuron_index] > current)
      &&(!max_value.compare_exchange_weak(current,neuron_data[neuron_index]))
    )current = max_value;

    current = expsum;
    while(!expsum.compare_exchange_weak(current, (current + std::exp(neuron_data[neuron_index]))))
      current = expsum;
  }, thread_index);

  double used_max_value = max_value;
  double used_expsum = expsum / std::exp(max_value);
  /*!Note: to make the softmax function numerically stable, the maximum value is subtracted from all values, and the sum is corrected for that.
   * Because the maximum value is not known during the calculation of the sum, it needs to be corrected by dividing the sum with exp(max(x)).
   * This is possible, because expq(x-c) = exp(x) / exp(c) for every element in the sum.
   */
  used_expsum = std::max(used_expsum, std::numeric_limits<double>::epsilon());
  /*!Note: Since no Neuron will be involved in the execution twice, no need for a mutual exclusive lock or atomics */
  execute_in_paralell_for(relevant_neurons,[&neuron_data, &used_max_value, &used_expsum](std::uint32_t neuron_index){
    neuron_data[neuron_index] = std::exp(neuron_data[neuron_index] - used_max_value) / used_expsum;
  }, thread_index);
}

void RafkoNetworkFeature::execute_dropout(
  NeuronDataProxy neuron_data,
  const rafko_mainframe::RafkoSettings& settings,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::uint32_t thread_index
) const{
  /*!Note: Since no Neuron will be involved in the execution twice, no need for a mutual exclusive lock or atomics */
  execute_in_paralell_for(relevant_neurons,[&neuron_data, &settings](std::uint32_t neuron_index){
    if((settings.get_dropout_probability()*(100.0)) >= static_cast<double>(rand()%100 + 1u)){
      neuron_data[neuron_index] = 0.0;
    }
  }, thread_index);
}

double RafkoNetworkFeature::calculate_l1_regularization(
  const RafkoNet& network,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::uint32_t thread_index
) const{
  std::atomic<double> error_value = 0.0;
  execute_in_paralell_for(relevant_neurons,[&error_value, &network](std::uint32_t neuron_index){
    SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
    [&error_value, &network](std::uint32_t weight_index){
      double current = error_value;
      while(!error_value.compare_exchange_weak(current, (error_value + std::abs(network.weight_table(weight_index)))))
        current = error_value;
    });
  }, thread_index);
  return static_cast<double>(error_value);
}

double RafkoNetworkFeature::calculate_l2_regularization(
  const RafkoNet& network,
  const google::protobuf::RepeatedPtrField<IndexSynapseInterval>& relevant_neurons,
  std::uint32_t thread_index
) const{
  std::atomic<double> error_value = 0.0;
  execute_in_paralell_for(relevant_neurons,[&error_value, &network](std::uint32_t neuron_index){
    SynapseIterator<>::iterate(network.neuron_array(neuron_index).input_weights(),
    [&error_value, &network](std::uint32_t weight_index){
      double current = error_value;
      while(!error_value.compare_exchange_weak(
        current, ( error_value + std::pow(network.weight_table(weight_index), (2.0)) )
      ))current = error_value;
    });
  }, thread_index);
  return static_cast<double>(error_value);
}

#if(RAFKO_USES_OPENCL)
void RafkoNetworkFeature::add_default_kernel_code_to(
  std::string& operations, const FeatureGroup& feature_group,
  const rafko_mainframe::RafkoSettings& settings, const Solution& solution,
  std::string input_array, std::string input_start_index,
  std::string output_array, std::string output_start_index,
  bool declare_locals
){
  std::vector<std::uint32_t> relevant_index_values;
  std::unordered_map<std::uint32_t, std::uint32_t> map_neurons_to_partials;
  std::unordered_map<std::uint32_t, std::uint32_t> map_weight_starts_to_partials;
  std::unordered_map<std::uint32_t, std::uint32_t> map_weight_sypase_start_to_neurons;
  switch(feature_group.feature()){
    case neuron_group_feature_softmax:
    case neuron_group_feature_dropout_regularization:{
      SynapseIterator<>::iterate(feature_group.relevant_neurons(),[&relevant_index_values](std::uint32_t index){
        relevant_index_values.push_back(index);
      });
    } break;
    case neuron_group_feature_l1_regularization:
    case neuron_group_feature_l2_regularization: {
      SynapseIterator<>::iterate(feature_group.relevant_neurons(), [&](std::uint32_t neuron_index){
        /* find neuron inside the partial solutions */
        std::uint32_t partial_index = rafko_gym::RafkoWeightAdapter::get_relevant_partial_index_for(
          neuron_index, solution, map_neurons_to_partials
        );
        const PartialSolution& partial = solution.partial_solutions(partial_index);
        std::uint32_t weight_start_in_partial = rafko_gym::RafkoWeightAdapter::get_device_weight_table_start_for(
          partial_index, solution, map_weight_starts_to_partials
        );
        std::uint32_t inner_neuron_index = (neuron_index - solution.partial_solutions(partial_index).output_data().starts());
        std::uint32_t weight_start_synapse = rafko_gym::RafkoWeightAdapter::get_weight_synapse_start_index_in_partial(
          neuron_index, partial, map_weight_sypase_start_to_neurons
        );
        SynapseIterator<>::iterate(partial.weight_indices(), [&relevant_index_values, weight_start_in_partial](std::uint32_t weight_index){
          relevant_index_values.push_back(weight_start_in_partial + weight_index);
        }, weight_start_synapse, partial.weight_synapse_number(inner_neuron_index));
      });
    } break;
    default: break; /* unknown functionality should yield no relevant index values */
  }
  operations += generate_kernel_code(
    settings, feature_group.feature(), relevant_index_values,
    input_array, input_start_index, output_array, output_start_index,
    declare_locals
  );
}

std::string RafkoNetworkFeature::generate_kernel_code(
  const rafko_mainframe::RafkoSettings& settings, Neuron_group_features feature,
  const std::vector<std::uint32_t>& relevant_index_values,
  std::string input_array, std::string input_start_index,
  std::string output_array, std::string output_start_index,
  bool declare_locals
){
  std::string operations = "";
  switch(feature){
    case neuron_group_feature_softmax:
      add_softmax_kernel_to(operations, relevant_index_values, output_array, output_start_index, declare_locals);
      break;
    case neuron_group_feature_dropout_regularization:
      add_dropout_kernel_to(
        operations, settings, relevant_index_values,
        output_array, output_start_index, declare_locals
      );
      break;
    case neuron_group_feature_l1_regularization:
      add_l1_kernel_to(
        operations, relevant_index_values,
        input_array, input_start_index, output_array, output_start_index, declare_locals
      );
      break;
    case neuron_group_feature_l2_regularization:
      add_l2_kernel_to(
        operations, relevant_index_values,
        input_array, input_start_index, output_array, output_start_index, declare_locals
      );
      break;
    default: break;
  }
  return operations;
}

void RafkoNetworkFeature::add_softmax_kernel_to(
  std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
  std::string neuron_data_array, std::string neuron_data_start_index, bool declare_locals
){
  std::regex index_regex("==index==");

  std::function<std::string(std::uint32_t)> index_string = [neuron_data_start_index](std::uint32_t index){
    return (neuron_data_start_index + " + " + std::to_string(index));
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
  std::for_each(relevant_index_values.begin(), relevant_index_values.end(),
  [&operations, &index_string, index_regex](std::uint32_t neuron_index){
    operations += rafko_utilities::replace_all_in_string(R"(max_val = max(max_val, ==neuron_data_array==[==index==]);
       exp_sum += exp(==neuron_data_array==[==index==]);
     )", index_regex, index_string(neuron_index));
  });

  /* epsilon guard */
  operations += "exp_sum = max(exp_sum, DBL_EPSILON);\n";

  /* apply softmax */
  std::for_each(relevant_index_values.begin(), relevant_index_values.end(),
  [&operations, &index_string, index_regex](std::uint32_t neuron_index){
    operations += rafko_utilities::replace_all_in_string(
      "==neuron_data_array==[==index==] = exp(==neuron_data_array==[==index==] - max_val) / (exp_sum / exp(max_val));\n\n",
      index_regex, index_string(neuron_index)
    );
  });
  operations = rafko_utilities::replace_all_in_string(operations, std::regex("==neuron_data_array=="), neuron_data_array);
  operations += "\n\n";
}

void RafkoNetworkFeature::add_dropout_kernel_to(
  std::string& operations, const rafko_mainframe::RafkoSettings& settings,
  const std::vector<std::uint32_t>& relevant_index_values,
  std::string neuron_data_array, std::string neuron_data_start_index, bool declare_locals
){
  std::string dropout_operations = "";
  std::string locals = "";
  if(declare_locals){
    locals += R"(
      uint dropout_seed = 0;
      dropout_seed = ==neuron_data_array==[get_random_number(==range==, &dropout_seed)];
    )";
  }else{
    locals += "dropout_seed = ==neuron_data_array==[get_random_number(==range==, &dropout_seed)];\n";
  }
  std::uint32_t number_of_neurons = 0u;
  std::for_each(relevant_index_values.begin(), relevant_index_values.end(),
  [&dropout_operations, neuron_data_start_index, &settings, &number_of_neurons, &neuron_data_array](std::uint32_t neuron_index){
     std::string this_op = R"(
      if((==param== * 100.0) >= (get_random_number(100, &dropout_seed) + 1)){
        ==neuron_data_array==[==neuron_data_start== + ==index==] = 0.0;
      }
    )";
    this_op = std::regex_replace(this_op, std::regex("==param=="), std::to_string(settings.get_dropout_probability()));
    this_op = std::regex_replace(this_op, std::regex("==neuron_data_start=="), neuron_data_start_index);
    this_op = std::regex_replace(this_op, std::regex("==index=="), std::to_string(neuron_index));
    this_op = std::regex_replace(this_op, std::regex("==neuron_data_array=="), neuron_data_array);
    dropout_operations += this_op;
    ++number_of_neurons;
  });
  dropout_operations = std::regex_replace(dropout_operations, std::regex("==range=="), std::to_string(number_of_neurons));
  operations += locals + dropout_operations;
}

void RafkoNetworkFeature::add_l1_kernel_to(
  std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
  std::string weight_array, std::string weight_start_index,
  std::string output_array, std::string output_start_index, bool declare_locals
){
  add_lx_kernel_to(
    operations, [](std::string input){ return "fabs(" + input + ")"; }, "l1_error",
    relevant_index_values,
    weight_array, weight_start_index, output_array, output_start_index, declare_locals
  );
}

void RafkoNetworkFeature::add_l2_kernel_to(
  std::string& operations, const std::vector<std::uint32_t>& relevant_index_values,
  std::string weight_array, std::string weight_start_index,
  std::string output_array, std::string output_start_index, bool declare_locals
){
  add_lx_kernel_to(
    operations, [](std::string input){ return "pow(" + input + ", 2.0)"; }, "l2_error",
    relevant_index_values,
    weight_array, weight_start_index, output_array, output_start_index, declare_locals
  );
}

void RafkoNetworkFeature::add_lx_kernel_to(
  std::string& operations, std::function<std::string(std::string)>&& lx, std::string local_name,
  const std::vector<std::uint32_t>& relevant_index_values,
  std::string weight_array, std::string weight_start_index,
  std::string output_array, std::string output_start_index,
  bool declare_locals
){
  {
    std::lock_guard<std::mutex> my_lock(m_featureCacheMutex);
    ++m_lxFeatureCalled;
    /*!Note: if the counter would increase at the end of the function, the index 0
     * would not be wasted, but this way the function is also thread-safe
     */
  }

  std::string feature_helpers;
  if(declare_locals){
    feature_helpers = R"(
      double ==local_name== = 0.0;
    )";
  }

  feature_helpers += R"(
    const int ==index_var_name==[==feature_weight_number==] = {
      ==index_values==
    };
  )";
  std::string index_list;
  std::for_each(relevant_index_values.begin(), relevant_index_values.end(),
  [&](std::uint32_t relevant_index_value){
    index_list += "(" + weight_start_index + "+" + std::to_string(relevant_index_value) + "),";
  });
  index_list.pop_back();
  std::string index_size_variable = "feature_lx_" + std::to_string(m_lxFeatureCalled) + "_index_values";
  std::string index_variable = "feature_lx_" + std::to_string(m_lxFeatureCalled) + "_index_values";

  /* add feature calculations */
  std::string feature_calculations = feature_helpers + R"(
    if(get_global_id(0) < ==feature_weight_number==){
      const int weights_in_one_thread = 1 + ( ==feature_weight_number==/(int)(get_global_size(0)) );
      const int execution_start_index = weights_in_one_thread * get_global_id(0);
      const int weights_in_this_thread = min(
        weights_in_one_thread, max(0, (==feature_weight_number== - execution_start_index))
      );

      ==local_name== = 0.0;
      for(
        int w_i = 0;
        (
          (w_i < weights_in_this_thread)
          &&((execution_start_index + w_i) < ==feature_weight_number==)
        );
        ++w_i
      ){
        ==local_name== += ==lx==;
      }

      AtomicAdd(&==output_array==[==output_start_index==], ==local_name==);
    }/*if(evaluating network)*/
  )";

  /* Replace helper variables */
  feature_calculations = std::regex_replace(feature_calculations, std::regex("==output_array=="), output_array);
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==lx=="),
    lx("==weight_array==[==index_var_name==[execution_start_index + w_i]]")
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==index_values=="), index_list
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==index_var_name=="), index_variable
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==feature_weight_number=="),
    std::to_string(relevant_index_values.size())
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==output_start_index=="), output_start_index
  );
  feature_calculations = rafko_utilities::replace_all_in_string(
    feature_calculations, std::regex("==local_name=="), local_name
  );
  feature_calculations = std::regex_replace(feature_calculations, std::regex("==output_array=="), output_array);
  feature_calculations = std::regex_replace(feature_calculations, std::regex("==weight_array=="), weight_array);
  operations += feature_calculations;
}

#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net */
