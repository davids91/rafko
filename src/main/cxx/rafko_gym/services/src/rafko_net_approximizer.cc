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

#include "rafko_gym/services/rafko_net_approximizer.h"

#include <algorithm>

#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

void RafkoNetApproximizer::collect_approximates_from_weight_gradients(){
  double gradient_overview = get_gradient_for_all_weights() * context.expose_settings().get_learning_rate(iteration);
  double greatest_weight_value = (0.0);
  std::vector<double>& tmp_weight_gradients = tmp_data_pool.reserve_buffer(context.expose_network().weight_table_size());
  used_weight_filter = weight_filter;
  for(std::uint32_t weight_index = 0; weight_index < tmp_weight_gradients.size(); ++weight_index){
    if( weight_exclude_chance_filter[weight_index] >= (static_cast<double>(rand()%100 + 1)/100.0) ){
      used_weight_filter[weight_index] = 0.0;
    }
    if(0.0 != used_weight_filter[weight_index]){
      tmp_weight_gradients[weight_index] = get_single_weight_gradient(weight_index) * used_weight_filter[weight_index];
      if(greatest_weight_value < std::abs(tmp_weight_gradients[weight_index]))
        greatest_weight_value = std::abs(tmp_weight_gradients[weight_index]);
    }else{
      tmp_weight_gradients[weight_index] = 0.0;
    }
  }
  for(std::uint32_t weight_index = 0; weight_index < tmp_weight_gradients.size(); ++weight_index){
    if(0.0 != used_weight_filter[weight_index]){
      tmp_weight_gradients[weight_index] = ( /* Gradients normalized by the biggest value */
        ( tmp_weight_gradients[weight_index] + gradient_overview ) / (greatest_weight_value + std::abs(gradient_overview))
      ); /*!Note: the biggest value in the weight gradients should be at most 1.0 after normalization,
          * so dividing by 1.0 + gradient_overview should normalize the offseted gradients
          */
      tmp_weight_gradients[weight_index] *= context.expose_settings().get_learning_rate(iteration);
    }
  }

  convert_direction_to_gradient(tmp_weight_gradients,true);
  tmp_data_pool.release_buffer(tmp_weight_gradients);
  ++iteration;
}

void RafkoNetApproximizer::convert_direction_to_gradient(std::vector<double>& direction, bool save_to_fragment){
  if(context.expose_network().weight_table_size() == static_cast<std::int32_t>(direction.size())){
    double error_negative_direction;
    double error_positive_direction;
    std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(context.expose_network().weight_table_size());

    for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index){
      if(0.0 != used_weight_filter[weight_index]){
        tmp_weight_table[weight_index] = context.expose_network().weight_table(weight_index) - direction[weight_index];
      }else{
        tmp_weight_table[weight_index] = context.expose_network().weight_table(weight_index);
      }
    } /* apply the direction on which network approximation shall be done */
    context.set_network_weights(tmp_weight_table);

    /* see the error values at the negative end of the current direction */
    context.push_state();
    error_negative_direction = -stochastic_evaluation();
    context.pop_state();

    /* see the error values at the positive end of the current direction */
    for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index)
      tmp_weight_table[weight_index] += direction[weight_index] * (2.0);
    context.set_network_weights(tmp_weight_table);

    if(!save_to_fragment)context.push_state();
    error_positive_direction = -stochastic_evaluation();
    if(!save_to_fragment)context.pop_state(); /* Restore train set to previous error state, decide if dampening is needed */

    /* collect the fragment, revert weight changes */
    double max_error = std::max(error_positive_direction,error_negative_direction);

    epsilon_addition = max_error / (max_error - std::min(error_positive_direction,error_negative_direction));
    /*!Note: In case the error delta between minimum and maximum error is small relative to the maximum error,
     * the gradient seems to be flat enough to slow training down, so
     * weight epsilon will be increased during approximation to help explore surrounding settings more.
     */
    std::vector<double>& tmp_weight_gradients = tmp_data_pool.reserve_buffer(context.expose_network().weight_table_size());
    for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index){
      if(0.0 != used_weight_filter[weight_index]){
        tmp_weight_gradients[weight_index] = ( ( error_positive_direction - error_negative_direction ) / (max_error) );
         if(save_to_fragment)add_to_fragment( weight_index, (tmp_weight_gradients[weight_index] * direction[weight_index]) );
        tmp_weight_table[weight_index] -= direction[weight_index];
      }
    }
    context.set_network_weights(tmp_weight_table);
    tmp_data_pool.release_buffer(tmp_weight_gradients);
    tmp_data_pool.release_buffer(tmp_weight_table);
  }else throw std::runtime_error("Incompatible direction given to approximate for!");
}

double RafkoNetApproximizer::get_single_weight_gradient(std::uint32_t weight_index){
  double gradient;
  const double current_epsilon = context.expose_settings().get_sqrt_epsilon() * epsilon_addition;
  const double current_epsilon_double = current_epsilon * (2.0);

  /* Push the choosen weight in one direction */
  context.set_network_weight( weight_index, (context.expose_network().weight_table(weight_index) + current_epsilon) );

  context.push_state(); /* Approximate the modified weights gradient */
  gradient = -stochastic_evaluation();
  context.pop_state();

  /* Push the selected weight in other direction */
  context.set_network_weight( weight_index, (context.expose_network().weight_table(weight_index) - current_epsilon_double) );
  context.push_state(); /* Approximate the newly modified weights gradient */
  double new_error_state = -stochastic_evaluation();
  context.pop_state();

  /* Calculate the gradient and revert weight modification and the error state with it */
  // if(0 < std::abs((gradient - new_error_state)))
  //   std::cout << "w[" << weight_index << "]: " << std::abs(gradient - new_error_state) << ";";
  gradient = -(gradient - new_error_state) * (current_epsilon_double);
  context.set_network_weight( weight_index, (context.expose_network().weight_table(weight_index) + current_epsilon) );

  return gradient;
}

double RafkoNetApproximizer::get_gradient_for_all_weights(){
  double gradient;
  double error_negative_direction;
  double error_positive_direction;
  const double current_epsilon = context.expose_settings().get_sqrt_epsilon();
  const double current_epsilon_double = current_epsilon * (2.0);
  std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(context.expose_network().weight_table_size());

  for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index){
    tmp_weight_table[weight_index] = context.expose_network().weight_table(weight_index) + current_epsilon;
  } /* Push every weight in a positive epsilon direction */
  context.set_network_weights(tmp_weight_table);

  context.push_state(); /* Approximate the modified weights gradient */
  error_positive_direction = -stochastic_evaluation();
  context.pop_state();

  for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index){
    tmp_weight_table[weight_index] -= current_epsilon_double;
  } /* Push the weights to the other direction */
  context.set_network_weights(tmp_weight_table);
  context.push_state(); /* Approximate the newly modified weights gradient */
  error_negative_direction = -stochastic_evaluation();
  context.pop_state();

  gradient = -(error_positive_direction - error_negative_direction) * (current_epsilon_double);
  for(std::uint32_t weight_index = 0; weight_index < tmp_weight_table.size(); ++weight_index){
    tmp_weight_table[weight_index] += current_epsilon;
  } /* Revert weight modifications and the error state with it */
  context.set_network_weights(tmp_weight_table);
  tmp_data_pool.release_buffer(tmp_weight_table);
  return (gradient);
}

void RafkoNetApproximizer::add_to_fragment(std::uint32_t weight_index, double gradient_fragment_value){
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

void RafkoNetApproximizer::apply_weight_vector_delta(){
  std::uint32_t fragment_value_index = 0;
  std::vector<double>& tmp_weight_table = tmp_data_pool.reserve_buffer(context.expose_network().weight_table_size());
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

  context.apply_weight_update(tmp_weight_table);
  tmp_data_pool.release_buffer(tmp_weight_table);
  gradient_fragment = NetworkWeightVectorDelta();
}

} /* namespace rafko_gym */
