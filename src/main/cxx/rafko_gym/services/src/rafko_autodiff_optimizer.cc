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
#include "rafko_gym/services/rafko_autodiff_optimizer.h"

#include <limits>
#include <deque>

#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/neuron_router.h"
#include "rafko_gym/services/rafko_backprop_network_input_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_bias_operation.h"
#include "rafko_gym/services/rafko_backprop_neuron_input_operation.h"
#include "rafko_gym/services/rafko_backprop_transfer_fn_operation.h"
#include "rafko_gym/services/rafko_backprop_objective_operation.h"
#include "rafko_gym/services/rafko_backprop_weight_reg_operation.h"
#include "rafko_gym/services/rafko_backprop_solution_feature_operation.h"

namespace rafko_gym{

std::uint32_t RafkoAutodiffOptimizer::build_without_data(std::shared_ptr<RafkoObjective> objective){
  RFASSERT_SCOPE(AUTODIFF_BUILD);
  RFASSERT(unplaced_spikes.empty());
  RFASSERT(spike_solves_feature_map.empty());
  std::uint32_t weight_relevant_operation_count;
  operations.clear();
  neuron_spike_to_operation_map->clear();
  data.reset();
  if(training_evaluator){
    training_evaluator->set_objective(objective);
  }

  /*!Note: other components depend on the output objectives being the first operations in the array. */
  for(std::uint32_t output_index = 0; output_index < network.output_neuron_number(); ++output_index){
    operations.push_back(std::make_shared<RafkoBackpropObjectiveOperation>(
      data, network, *objective, operations.size(),
      output_index, environment->get_number_of_label_samples()
    ));
    RFASSERT_LOG(
      "operation[{}]: {} for output {} ",
      operations.size()-1, Autodiff_operations_Name(ad_operation_objective_d), output_index
    );
  }

  /* handle the group feature related operations, upload performance related feature group operations */
  std::uint32_t feature_group_index = 0u;
  for(const rafko_net::FeatureGroup& feature_group : network.neuron_group_features()){
    if(rafko_net::NeuronInfo::is_feature_relevant_to_performance(feature_group.feature()) ){
      operations.push_back(std::make_shared<RafkoBackpropWeightRegOperation>(
        settings, data, network, operations.size(), feature_group
      ));
      /*!Note: weight_relevant_operation_count counts on the placed items into the operation array here */
      RFASSERT_LOG(
        "operation[{}]: {} for feature_group[{}]",
        operations.size()-1, Autodiff_operations_Name(ad_operation_network_weight_regularization_feature),
        feature_group_index
      );
    }
    ++feature_group_index;
  }
  weight_relevant_operation_count = operations.size();

  /* Collect the Neuron subsets to determine the order of placement */
  rafko_net::NeuronRouter neuron_router(network);
  std::vector<std::deque<std::uint32_t>> neuron_subsets;
  bool strict_mode = false;
  while(!neuron_router.finished()){
    neuron_router.collect_subset(settings.get_max_solve_threads(), settings.get_device_max_megabytes(), strict_mode);
    neuron_subsets.insert( neuron_subsets.begin(),  {neuron_router.get_subset().rbegin(), neuron_router.get_subset().rend()} );
    /*!Note: new subsets are inserted into the beginning of the array, so that the Neurons depending on everything
     * will be place to the beginning of the operations array, as it is being executed from the end towards the beginning.
     * The deque is added with the reverse iterators, because in non-strict mode order matters, so the neuron at the end
     * might depend on the other neurons, so it must be placed before them, in strict mode the order doesn't matter.
     */

    for(const std::uint32_t& neuron_index : neuron_router.get_subset()){ /* confirm each Neuron as processed, and store the resultin solved feature groups */
      std::vector<std::uint32_t> solved_features = neuron_router.confirm_first_subset_element_processed(neuron_index);
      for(std::uint32_t feature_group_index : solved_features){
        if( rafko_net::NeuronInfo::is_feature_relevant_to_solution(network.neuron_group_features(feature_group_index).feature()) ){
          spike_solves_feature_map.insert({neuron_index, feature_group_index});
        }
      }/*for(each solved feature group index)*/
    }/*for(each neuron_index in the subset)*/
    strict_mode = false; /* Strict mode should only run at the first subset collection */
  }/*while(neuron_router is finished)*/

  RFASSERT_LOGV2( neuron_subsets, "Subset array:");

  /* Place one subset of Neurons */
  std::uint32_t done_index = 0;
  while(0u < neuron_subsets.size()){
    for(std::uint32_t neuron_index : *neuron_subsets.begin()){
      auto found_feature = spike_solves_feature_map.find(neuron_index);
      if(found_feature != spike_solves_feature_map.end()){
        operations.push_back( std::make_shared<RafkoBackPropSolutionFeatureOperation>(
          data, network, operations.size(), settings, network.neuron_group_features(found_feature->second),
          execution_threads, neuron_spike_to_operation_map
        ));
        RFASSERT_LOG(
          "operation[{}]:  {} for feature_group[{}], triggered by Neuron[]",
          operations.size(), Neuron_group_features_Name(network.neuron_group_features(found_feature->second).feature()),
          found_feature->second, found_feature->first
        );
      }

      place_spike_to_operations(neuron_index);

      /* Upload dependencies for every operation until every dependency is registered */
      while(done_index < operations.size()){
        if(!operations[done_index]->are_dependencies_registered()){
          DependencyRequest request = operations[done_index]->upload_dependencies_to_operations();
          if(request.has_value()){
            auto& [parameters, dependency_register] = request.value();
            std::vector<std::shared_ptr<RafkoBackpropagationOperation>> new_dependencies;
            for(const DependencyParameter& parameter : parameters){
              new_dependencies.push_back(push_dependency(parameter));
            }
            dependency_register(new_dependencies);
          }
        }
        ++done_index;
      }/*while(done_index < operations.size())*/
    }/*for(every neuron_index in the collected subset begin)*/
    neuron_subsets.erase(neuron_subsets.begin());
  }/*while(subsets remain)*/

  RFASSERT_LOG("Spike map:");
  #if(RAFKO_USES_ASSERTLOGS)
  for(const auto& [neuron_index, operation_index] : *neuron_spike_to_operation_map){
    RFASSERT_LOG("Neuron[{}] --> Operation[{}]", neuron_index, operation_index);
  }
  #endif/*(RAFKO_USES_ASSERTLOGS)*/
  RFASSERT_LOG("============================");

  return weight_relevant_operation_count;
}

void RafkoAutodiffOptimizer::calculate_value(const std::vector<double>& network_input){
  for(std::int32_t operation_index = operations.size() - 1; operation_index >= 0; --operation_index){
    operations[operation_index]->calculate_value(network_input);
  }
}

void RafkoAutodiffOptimizer::calculate_derivative(
  const std::vector<double>& network_input, const std::vector<double>& label_data
){
  execution_threads[0]->start_and_block([this, &network_input, &label_data](std::uint32_t thread_index){
    const std::int32_t weights_in_one_thread = 1 + (network.weight_table_size() / execution_threads[0]->get_number_of_threads());
    const std::int32_t weight_start_in_thread = (weights_in_one_thread * thread_index);
    const std::int32_t weights_to_do_in_this_thread = std::min(
      weights_in_one_thread, (network.weight_table_size() - weight_start_in_thread)
    );
    for(
      std::int32_t weight_index = weight_start_in_thread;
      weight_index < (weight_start_in_thread + weights_to_do_in_this_thread);
       ++weight_index
    ){
      for(std::int32_t operation_index = operations.size() - 1; operation_index >= 0; --operation_index)
        operations[operation_index]->calculate_derivative(static_cast<std::uint32_t>(weight_index), network_input, label_data);
    }
  });
}

void RafkoAutodiffOptimizer::calculate(BackpropDataBufferRange network_input, BackpropDataBufferRange label_data){
  RFASSERT_SCOPE(AUTODIFF_CALCULATE);
  for(std::uint32_t run_index = 0; run_index < network_input.size(); ++run_index){
    data.step();
    calculate_value(network_input[run_index]);
    calculate_derivative(network_input[run_index], label_data[run_index]);
  }
}

void RafkoAutodiffOptimizer::update_context_errors(){
  if( (training_evaluator) && (0 == (iteration%settings.get_tolerance_loop_value())) ){
    training_evaluator->refresh_solution_weights();
    last_training_error = training_evaluator->stochastic_evaluation();
  }
  if(
    (test_evaluator)
    &&(
      (0 == (iteration%settings.get_tolerance_loop_value()))
      ||(
        (training_evaluator)
        &&((last_training_error * settings.get_delta()) < std::abs(last_training_error - last_testing_error))
      )
    )
  ){
    test_evaluator->refresh_solution_weights();
    last_testing_error = test_evaluator->stochastic_evaluation();
  }
}

void RafkoAutodiffOptimizer::iterate(){
  RFASSERT(static_cast<bool>(environment));
  std::uint32_t sequence_start_index = (rand()%(environment->get_number_of_sequences() - used_minibatch_size + 1));
  std::uint32_t start_index_inside_sequence = (rand()%( /* If the memory is truncated for the training.. */
    environment->get_sequence_size() - used_sequence_truncation + 1u /* ..not all result output values are evaluated.. */
  )); /* ..only settings.get_memory_truncation(), starting at a random index inside bounds */

  for(std::uint32_t sequence_index = sequence_start_index; sequence_index < used_minibatch_size; ++sequence_index){
    std::uint32_t raw_inputs_index = sequence_index * (environment->get_sequence_size() + environment->get_prefill_inputs_number());
    std::uint32_t raw_labels_index = sequence_index * environment->get_sequence_size();

    /* Evaluate the current sequence step by step */
    reset();
    for(std::uint32_t prefill_iterator = 0; prefill_iterator < environment->get_prefill_inputs_number(); ++prefill_iterator){
      data.step();
      calculate_value(environment->get_input_sample(raw_inputs_index));
      ++raw_inputs_index;
    } /* The first few inputs are there to set an initial state to the network */

    /* Solve the data and store the result after the inital "prefill" */
    for(std::uint32_t sequence_index = 0; sequence_index < environment->get_sequence_size(); ++sequence_index){
      data.step();
      data.set_weight_derivative_update( /* Add to the relevant derivatives only when truncation parameters match */
        (sequence_index >= start_index_inside_sequence)
        &&(sequence_index < (start_index_inside_sequence + used_sequence_truncation))
      );
      calculate_value(environment->get_input_sample(raw_inputs_index));
      calculate_derivative( environment->get_input_sample(raw_inputs_index), environment->get_label_sample(raw_labels_index) );
      ++raw_inputs_index;
      ++raw_labels_index;
    }/*for(relevant sequences)*/
  }

  std::fill(tmp_avg_derivatives.begin(), tmp_avg_derivatives.end(), 0.0);
  for(
    std::uint32_t past_sequence_index = start_index_inside_sequence;
    past_sequence_index < (start_index_inside_sequence + used_sequence_truncation);
    ++past_sequence_index
  ){
    const std::vector<double>& sequence_derivative = data.get_average_derivative()
      .get_element(past_sequence_index);
    std::transform(
      sequence_derivative.begin(), sequence_derivative.end(),
      tmp_avg_derivatives.begin(), tmp_avg_derivatives.begin(),
      [](const double& a, const double& b){ return (a+b)/2.0; }
    );
  }

  apply_weight_update(tmp_avg_derivatives);
  ++iteration;

  update_context_errors();
}

double RafkoAutodiffOptimizer::get_avg_gradient(std::uint32_t d_w_index){
  double sum = 0.0;
  double count = 0.0;
  for(std::uint32_t past_index = 0u; past_index < network.memory_size(); ++past_index){
    sum += data.get_average_derivative(past_index, d_w_index);
    count += 1.0;
  }
  return sum / count;
}

std::shared_ptr<RafkoBackpropagationOperation> RafkoAutodiffOptimizer::place_spike_to_operations(std::uint32_t neuron_index){
  /* find the Spike index in the not yet placed Neuron spikes */
  auto found_element = unplaced_spikes.find(neuron_index);
  if(found_element != unplaced_spikes.end()){
    found_element->second->set_operation_index(operations.size());
    operations.push_back(found_element->second);
    unplaced_spikes.erase(found_element);
    RFASSERT_LOG(
      "operation[{}]:  Neuron[{}] {} inserted from unplaced spikes",
      operations.size() - 1u, neuron_index, Autodiff_operations_Name(ad_operation_neuron_spike_d)
    );
  }else{
    operations.push_back(std::make_shared<RafkoBackpropSpikeFnOperation>(
      data, network, operations.size(), neuron_index
    ));
    RFASSERT_LOG(
      "operation[{}]:  Neuron[{}] {} built, because not found elsewhere",
      operations.size() - 1u, neuron_index, Autodiff_operations_Name(ad_operation_neuron_spike_d)
    );
  }
  neuron_spike_to_operation_map->insert( {neuron_index, (operations.size() - 1u)} );
  return operations.back();
}

std::shared_ptr<RafkoBackpropagationOperation> RafkoAutodiffOptimizer::find_or_queue_spike(std::uint32_t neuron_index){
  { /* find the Spoike index in the already placed Neuron spikes */
    auto found_element = neuron_spike_to_operation_map->find(neuron_index);
    if(found_element != neuron_spike_to_operation_map->end())
      return operations[found_element->second];
  }

  { /* find the Spike index in the not yet placed Neuron spikes */
    auto found_element = unplaced_spikes.find(neuron_index);
    if(found_element != unplaced_spikes.end())
      return found_element->second;
  }

  /* Neuron index was not found, so add it to the unplaced spikes */
  auto insertion = unplaced_spikes.insert({ /* with a dummy operation index which is to be set in @place_spike */
    neuron_index, std::make_shared<RafkoBackpropSpikeFnOperation>(
      data, network, 0u/*operation index*/, neuron_index
    )
  });
  RFASSERT_LOG(
    "Neuron[{}] {} inserted into unplaced spikes",
    neuron_index, Autodiff_operations_Name(ad_operation_neuron_spike_d)
  );
  RFASSERT(std::get<1>(insertion));
  return std::get<0>(insertion)->second;
}

std::shared_ptr<RafkoBackpropagationOperation> RafkoAutodiffOptimizer::push_dependency(DependencyParameter arguments){
  switch(std::get<0>(arguments)){
    case ad_operation_neuron_spike_d:
      RFASSERT(1u == std::get<1>(arguments).size());
      return find_or_queue_spike(std::get<1>(arguments)[0]);
    case ad_operation_neuron_transfer_d:
      RFASSERT(1u == std::get<1>(arguments).size());
      RFASSERT_LOG(
        "operation[{}]: {} for Neuron[{}]",
        operations.size(), Autodiff_operations_Name(std::get<0>(arguments)), std::get<1>(arguments)[0]
      );
      return operations.emplace_back(new RafkoBackpropTransferFnOperation(
        data, network, operations.size(), std::get<1>(arguments)[0], settings
      ));
    case ad_operation_neuron_input_d:
      RFASSERT(2u == std::get<1>(arguments).size());
      RFASSERT_LOG(
        "operation[{}]: {} for Neuron[{}] input[{}]",
        operations.size(), Autodiff_operations_Name(std::get<0>(arguments)), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
      );
      return operations.emplace_back(new RafkoBackpropNeuronInputOperation(
        data, network, operations.size(), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
      ));
    case ad_operation_neuron_bias_d:
      RFASSERT(2u == std::get<1>(arguments).size());
      RFASSERT_LOG(
        "operation[{}]: {} for Neuron[{}] weight_input[{}] ( not weight index ) ",
        operations.size(), Autodiff_operations_Name(std::get<0>(arguments)), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
      );
      return operations.emplace_back(new RafkoBackpropNeuronBiasOperation(
        data, network, operations.size(), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
      ));
    case ad_operation_network_input_d:
      RFASSERT(3u == std::get<1>(arguments).size());
      RFASSERT_LOG(
        "operation[{}]: {} for Input[{}] weight_index[{}] (Neuron[{}])",
        operations.size(), Autodiff_operations_Name(std::get<0>(arguments)), std::get<1>(arguments)[0], std::get<1>(arguments)[1], std::get<1>(arguments)[2]
      );
      return operations.emplace_back(new RafkoBackpropNetworkInputOperation(
        data, network, operations.size(), std::get<1>(arguments)[0], std::get<1>(arguments)[1]
      ));
    break;
    case ad_operation_objective_d: /* Objective operations are placed manually to the beginning of the vector */
    case ad_operation_unknown:
    default: break;
  }
  return std::shared_ptr<RafkoBackpropagationOperation>();
}

} /* namespace rafko_gym */
