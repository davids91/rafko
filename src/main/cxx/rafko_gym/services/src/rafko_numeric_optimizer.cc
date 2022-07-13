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

#include "rafko_gym/services/rafko_numeric_optimizer.hpp"

#include <algorithm>

#include "rafko_net/services/synapse_iterator.hpp"

namespace rafko_gym{

void RafkoNumericOptimizer::collect_approximates_from_weight_gradients(){
  if(exclude_chance_sum < weight_exclude_chance_filter.size()){
    double greatest_gradient_value = settings.get_sqrt_epsilon();
    double used_weight_filter_sum = 0.0;
    std::mutex weight_stats_mutex;
    std::vector<double>& used_gradients = tmp_data_pool.reserve_buffer(contexts[0]->expose_network().weight_table_size());
    used_weight_filter = weight_filter;
    for(std::uint32_t weight_index = 0; weight_index < used_gradients.size(); weight_index += execution_threads.get_number_of_threads()){
      execution_threads.start_and_block(
      [this, weight_index, &used_gradients, &greatest_gradient_value, &weight_stats_mutex, &used_weight_filter_sum](std::uint32_t thread_index){
        const std::uint32_t actual_weight_index = (weight_index + thread_index);
        if(actual_weight_index < used_gradients.size()){
          if(
            (0 < exclude_chance_sum)
            &&(weight_exclude_chance_filter[actual_weight_index] >= (static_cast<double>(rand()%100 + 1)/100.0))
          ){
            used_weight_filter[actual_weight_index] = 0.0;
          }

          if(0.0 != used_weight_filter[actual_weight_index]){
            used_gradients[actual_weight_index] = get_single_weight_gradient(actual_weight_index, *contexts[thread_index]) * used_weight_filter[actual_weight_index];
            if(greatest_gradient_value < std::abs(used_gradients[actual_weight_index])){
              std::lock_guard<std::mutex> my_lock(weight_stats_mutex);
              greatest_gradient_value = std::abs(used_gradients[actual_weight_index]);
            }
            std::lock_guard<std::mutex> my_lock(weight_stats_mutex);
            used_weight_filter_sum += used_weight_filter[actual_weight_index];
          }else{
            used_gradients[actual_weight_index] = 0.0;
          }
        }/*if(actual_weight_index inside bounds)*/
      });
    }/*for(all weights)*/
    double gradient_overview = 0.0;
    double weight_filter_accumulate = 0.0;
    if(0.0 < used_weight_filter_sum){
      weight_filter_accumulate = std::accumulate(weight_filter.begin(), weight_filter.end(), 0.0);
    }
    if((0 < used_weight_filter_sum)||(0 < weight_filter_accumulate)){
      gradient_overview = get_gradient_for_all_weights() * contexts[0]->expose_settings().get_learning_rate(iteration);
    }
    if(0 < used_weight_filter_sum){ /* if any weights are included */
      for(std::uint32_t weight_index = 0; weight_index < used_gradients.size(); ++weight_index){
        if(0.0 != used_weight_filter[weight_index]){
          used_gradients[weight_index] = ( /* Gradients normalized by the biggest value */
            ( used_gradients[weight_index] + gradient_overview ) / (greatest_gradient_value + std::abs(gradient_overview))
          ); /*!Note: the biggest value in the weight gradients should be at most 1.0 after normalization,
              * so dividing by 1.0 + gradient_overview should normalize the offseted gradients
              */
          used_gradients[weight_index] *= contexts[0]->expose_settings().get_learning_rate(iteration);
        }
      }
      convert_direction_to_gradient(used_gradients,true);
    }else if(0 < weight_filter_accumulate){ /* if no weights were selected, but only by chance */
      std::uint32_t chosen_weight_index = rand()%weight_filter.size();
      while(0.0 == weight_filter[chosen_weight_index]){ /* find a suitable single random weight */
        chosen_weight_index = rand()%weight_filter.size();
      }
      /* approximize a single weight */
      used_gradients[chosen_weight_index] = get_single_weight_gradient(chosen_weight_index, *contexts[0]) * weight_filter[chosen_weight_index];
      used_gradients[chosen_weight_index] = (
        ( used_gradients[chosen_weight_index] + gradient_overview ) / (std::abs(used_gradients[chosen_weight_index]) + std::abs(gradient_overview))
      ) * contexts[0]->expose_settings().get_learning_rate(iteration);
      convert_direction_to_gradient(used_gradients,true);
    }
    tmp_data_pool.release_buffer(used_gradients);
  }/*if(at least some weights are not excluded)*/
  ++iteration;
}

void RafkoNumericOptimizer::convert_direction_to_gradient(std::vector<double>& direction, bool save_to_fragment){
  RFASSERT(contexts[0]->expose_network().weight_table_size() == static_cast<std::int32_t>(direction.size()));
  double error_negative_direction;
  double error_positive_direction;
  std::vector<double>& network_original_weights = tmp_data_pool.reserve_buffer(contexts[0]->expose_network().weight_table_size());
  std::vector<double>& negative_direction = tmp_data_pool.reserve_buffer(network_original_weights.size());
  std::vector<double>& tmp_weight_gradients = tmp_data_pool.reserve_buffer(network_original_weights.size());
  network_original_weights = {
    contexts[0]->expose_network().weight_table().begin(),
    contexts[0]->expose_network().weight_table().end()
  };

  if(2 <= execution_threads.get_number_of_threads()){
    execution_threads.start_and_block(
    [this, &direction, &negative_direction, &error_positive_direction, &error_negative_direction, &network_original_weights](std::uint32_t thread_index){
      std::vector<double>* used_direction;
      double* dir_error;
      if(thread_index == 0){
        used_direction = &direction;
        dir_error = &error_positive_direction;
      }else if(thread_index == 1){
        std::transform(direction.begin(),direction.end(), negative_direction.begin(),
        [](const double& direction_value){
          return -direction_value;
        });
        used_direction = &negative_direction;
        dir_error = &error_negative_direction;
      }

      if(thread_index < 2){
        *dir_error = get_error_from_direction(*contexts[thread_index], network_original_weights, *used_direction);
      }
    });
  }else{ /* Check directions sequentially */
    error_positive_direction = get_error_from_direction(*contexts[0], network_original_weights, direction);
    std::transform(direction.begin(),direction.end(), negative_direction.begin(),
    [](const double& direction_value){
      return -direction_value;
    });
    error_negative_direction = get_error_from_direction(*contexts[0], network_original_weights, negative_direction);
  }

  /* collect the fragment */
  double max_error = std::max(error_positive_direction,error_negative_direction);
  epsilon_addition = max_error / (max_error - std::min(error_positive_direction,error_negative_direction));
  /*!Note: In case the error delta between minimum and maximum error is small relative to the maximum error,
   * the gradient seems to be flat enough to slow training down, so
   * weight epsilon will be increased during approximation to help explore surrounding settings more.
   */
  for(std::uint32_t weight_index = 0; weight_index < network_original_weights.size(); ++weight_index){
    if(0.0 != used_weight_filter[weight_index]){
      tmp_weight_gradients[weight_index] = ( ( error_positive_direction - error_negative_direction ) / (max_error) );
       if(save_to_fragment)add_to_fragment( weight_index, (tmp_weight_gradients[weight_index] * direction[weight_index]) );
    }
  }

  tmp_data_pool.release_buffer(network_original_weights);
  tmp_data_pool.release_buffer(negative_direction);
  tmp_data_pool.release_buffer(tmp_weight_gradients);
}

double RafkoNumericOptimizer::get_single_weight_gradient(std::uint32_t weight_index, rafko_mainframe::RafkoContext& context){
  double gradient;
  const double current_epsilon = context.expose_settings().get_sqrt_epsilon() * epsilon_addition;

  { /* Push the choosen weight in one direction */
    std::lock_guard<std::mutex> my_lock(network_mutex);
    double w = context.expose_network().weight_table(weight_index);
    context.set_network_weight( weight_index, w + current_epsilon);
    context.expose_network().set_weight_table(weight_index, w);
  }

  context.push_state(); /* Approximate the modified weights gradient */
  gradient = -stochastic_evaluation(context);
  context.pop_state();

  { /* Push the selected weight in other direction */
    std::lock_guard<std::mutex> my_lock(network_mutex);
    double w = context.expose_network().weight_table(weight_index);
    context.set_network_weight( weight_index, w - current_epsilon);
    context.expose_network().set_weight_table(weight_index, w);
  }

  context.push_state(); /* Approximate the newly modified weights gradient */
  double new_error_state = -stochastic_evaluation(context);
  context.pop_state();

  gradient = -(gradient - new_error_state) * (2.0 * current_epsilon);

  return gradient;
}

double RafkoNumericOptimizer::get_error_from_direction(
  rafko_mainframe::RafkoContext& context,
  const std::vector<double>& network_original_weights,
  double direction
){
  double result_error;
  std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(network_original_weights.size());
  std::transform(network_original_weights.begin(), network_original_weights.end(), tmp_weight_table.begin(),
  [direction](const double& weight_value){ /* Push every weight in the given direction */
    return weight_value + direction;
  });

  { /* Push the choosen weight in the chosen direction */
    std::lock_guard<std::mutex> my_lock(network_mutex);
    context.set_network_weights(tmp_weight_table);
    *context.expose_network().mutable_weight_table() = {
      network_original_weights.begin(),
      network_original_weights.end()
    };
  }

  context.push_state(); /* Approximate the modified weights gradient */
  result_error = -stochastic_evaluation(context);
  context.pop_state();

  tmp_data_pool.release_buffer(tmp_weight_table);
  return result_error;
}

double RafkoNumericOptimizer::get_error_from_direction(
  rafko_mainframe::RafkoContext& context,
  const std::vector<double>& network_original_weights,
  const std::vector<double>& direction
){
  double result_error;
  std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(network_original_weights.size());
  std::uint32_t i = 0;
  std::transform(network_original_weights.begin(), network_original_weights.end(), tmp_weight_table.begin(),
  [&direction, &i](const double& weight_value){ /* Push every weight in the given direction */
    ++i;
    return weight_value + direction[i - 1u];
  });

  { /* Push the choosen weight in the chosen direction */
    std::lock_guard<std::mutex> my_lock(network_mutex);
    context.set_network_weights(tmp_weight_table);
    *context.expose_network().mutable_weight_table() = {
      network_original_weights.begin(),
      network_original_weights.end()
    };
  }

  context.push_state(); /* Approximate the modified weights gradient */
  result_error = -stochastic_evaluation(context);
  context.pop_state();

  tmp_data_pool.release_buffer(tmp_weight_table);
  return result_error;
}

double RafkoNumericOptimizer::get_gradient_for_all_weights(){
  double error_negative_direction;
  double error_positive_direction;
  const double current_epsilon = contexts[0]->expose_settings().get_sqrt_epsilon();
  const double current_epsilon_double = current_epsilon * (2.0);
  const std::vector<double> network_original_weights = {
    contexts[0]->expose_network().weight_table().begin(),
    contexts[0]->expose_network().weight_table().end()
  };

  if(2 <= execution_threads.get_number_of_threads()){
    execution_threads.start_and_block(
    [this, current_epsilon, &error_positive_direction, &error_negative_direction, &network_original_weights](std::uint32_t thread_index){
      double weight_addition;
      double* dir_error;
      if(thread_index == 0){
        weight_addition = current_epsilon;
        dir_error = &error_positive_direction;
      }else if(thread_index == 1){
        weight_addition = -current_epsilon;
        dir_error = &error_negative_direction;
      }

      if(thread_index < 2){
        *dir_error = get_error_from_direction(*contexts[thread_index], network_original_weights, weight_addition);
      }
    });
  }else{ /* Calculate the gradient sequentially */
    error_positive_direction = get_error_from_direction(*contexts[0], network_original_weights, current_epsilon);
    error_negative_direction = get_error_from_direction(*contexts[0], network_original_weights, -current_epsilon);
  }
  return -(error_positive_direction - error_negative_direction) * (current_epsilon_double);
}

void RafkoNumericOptimizer::add_to_fragment(std::uint32_t weight_index, double gradient_fragment_value){
  std::uint32_t values_index = 0;
  std::uint32_t values_index_target = gradient_fragment.values_size();
  std::uint32_t weight_synapse_index_target = gradient_fragment.weight_synapses_size();
  rafko_net::IndexSynapseInterval tmp_synapse_interval;

  for(std::uint32_t weight_syn_index = 0; static_cast<std::int32_t>(weight_syn_index) < gradient_fragment.weight_synapses_size(); ++weight_syn_index){
    if( /* If the weight synapse is at or in-between the first index before the start of the synapse.. */
      (
        ((0 < gradient_fragment.weight_synapses(weight_syn_index).starts())
        &&( gradient_fragment.weight_synapses(weight_syn_index).starts()-1) <= static_cast<std::int32_t>(weight_index) )
        ||( 0 == gradient_fragment.weight_synapses(weight_syn_index).starts() )
      )&&( /* ..and the one after the last index */
        (gradient_fragment.weight_synapses(weight_syn_index).starts() + gradient_fragment.weight_synapses(weight_syn_index).interval_size())
        >= weight_index
      )
    ){ /* current weight synapse is a sitable target to place the current fragment in */
      weight_synapse_index_target = weight_syn_index;
      values_index_target = values_index;
      break; /* Found a suitable synapse, no need to continue */
    }
    values_index += gradient_fragment.weight_synapses(weight_syn_index).interval_size();
  } /* Go through the synapses saving the last place */
  if(
    (0 == gradient_fragment.weight_synapses_size())
    ||(static_cast<std::int32_t>(weight_synapse_index_target) >= gradient_fragment.weight_synapses_size())
    ||(static_cast<std::int32_t>(values_index_target) >= gradient_fragment.values_size())
  ){
    gradient_fragment.add_values(gradient_fragment_value);
    tmp_synapse_interval.set_interval_size(1);
    tmp_synapse_interval.set_starts(weight_index);
    *gradient_fragment.add_weight_synapses() = tmp_synapse_interval;
  }else{
    const std::uint32_t synapse_starts = gradient_fragment.weight_synapses(weight_synapse_index_target).starts();
    const std::uint32_t synapse_size = gradient_fragment.weight_synapses(weight_synapse_index_target).interval_size();
    const std::uint32_t synapse_ends = synapse_starts + synapse_size;

    if(
      (0 < synapse_starts) /* Synapse doesn't start at 0 */
      &&((synapse_starts-1) == weight_index) /* And the weight index points to the first index before the synapse */
    ){
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_starts(synapse_starts - 1);
      insert_element_at_position(*gradient_fragment.mutable_values(),gradient_fragment_value,values_index_target);
    }else if(
      (synapse_starts <= weight_index)
      &&(synapse_ends > weight_index)
    ){ /* the index is inside the synapse */
      gradient_fragment.set_values(
        values_index_target + weight_index - synapse_starts,
        gradient_fragment.values(values_index_target + weight_index - synapse_starts) + gradient_fragment_value
      );
    }else{ /* The index is the first index after the synapse */
      gradient_fragment.mutable_weight_synapses(weight_synapse_index_target)->set_interval_size(synapse_size + 1);
      insert_element_at_position( *gradient_fragment.mutable_values(), gradient_fragment_value, (values_index_target + synapse_size) );
    }
  }
}

void RafkoNumericOptimizer::apply_weight_vector_delta(){
  std::uint32_t fragment_value_index = 0;
  std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(contexts[0]->expose_network().weight_table_size());
  std::fill(tmp_weight_table.begin(),tmp_weight_table.end(), (0.0));

  if(1 == gradient_fragment.weight_synapses_size()){
    std::copy(
      gradient_fragment.values().begin(),
      gradient_fragment.values().begin() + gradient_fragment.weight_synapses(0).interval_size(),
      tmp_weight_table.begin() + gradient_fragment.weight_synapses(0).starts()
    );
  }else{
    rafko_net::SynapseIterator<>::iterate(gradient_fragment.weight_synapses(), [&](std::int32_t weight_index){
      tmp_weight_table[weight_index] += gradient_fragment.values(fragment_value_index);
      ++fragment_value_index;
    });
  }

  contexts[0]->apply_weight_update(tmp_weight_table);
  tmp_data_pool.release_buffer(tmp_weight_table);
  gradient_fragment = NetworkWeightVectorDelta();
  /*!Note: This should help, but doesn't ..  contexts[0]->full_evaluation(); */
}

} /* namespace rafko_gym */
