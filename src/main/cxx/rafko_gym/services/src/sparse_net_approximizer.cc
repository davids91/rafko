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

#include "rafko_gym/services/sparse_net_approximizer.h"

#include <algorithm>

#include "gen/common.pb.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_gym{

using std::make_unique;
using std::lock_guard;
using std::ref;

using sparse_net_library::Synapse_iterator;
using sparse_net_library::Index_synapse_interval;

void Sparse_net_approximizer::collect_approximates_from_weight_gradients(void){
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  sdouble32 gradient_overview = get_gradient_for_all_weights() * service_context.get_step_size();
  sdouble32 greatest_weight_value = double_literal(0.0);
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = get_single_weight_gradient(weight_index);
    if(greatest_weight_value < std::abs(weight_gradients[weight_index]))
      greatest_weight_value = std::abs(weight_gradients[weight_index]);
  }
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = ( /* Gradients normalized by the biggest value */
      ( weight_gradients[weight_index] + gradient_overview ) / (greatest_weight_value + std::abs(gradient_overview))
    ); /*!Note: the biggest value in the weight gradients should be at most 1.0 after normalization,
        * so dividing by 1.0 + gradient_overview should normalize the offseted gradients
        */
    weight_gradients[weight_index] *= service_context.get_step_size();
  }
  convert_direction_to_gradient(weight_gradients,true);
  ++iteration;
}

void Sparse_net_approximizer::convert_direction_to_gradient(vector<sdouble32>& direction, bool save_to_fragment){
  if(net.weight_table_size() == static_cast<sint32>(direction.size())){
    sdouble32 dampening_value;
    sdouble32 error_negative_direction;
    sdouble32 error_positive_direction;
    sdouble32 weight_epsilon = double_literal(0.0);

    vector<sdouble32> weight_gradients(net.weight_table_size(), double_literal(0.0));
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - direction[weight_index]) );
      weight_epsilon += std::pow(direction[weight_index],double_literal(2.0)); /* Collect weight epsilon for both(positive and negative) direction */
    } /* apply the direction on which network approximation shall be done */
    weight_epsilon = std::sqrt(weight_epsilon);
    weight_updater->update_solution_with_weights(*net_solution);

    /* see the error values at the negative end of the current direction */
    environment.push_state();
    error_negative_direction = -environment.stochastic_evaluation(*solver, iteration);
    environment.pop_state();

    /* see the error values at the positive end of the current direction */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index)
      net.set_weight_table(weight_index, (net.weight_table(weight_index) + (direction[weight_index] * double_literal(2.0))) );
    weight_updater->update_solution_with_weights(*net_solution);

    if(!save_to_fragment)environment.push_state();
    error_positive_direction = -environment.stochastic_evaluation(*solver, iteration);
    if(!save_to_fragment)environment.pop_state(); /* Restore train set to previous error state, decide if dampening is needed */

    if( /* In case the initial error is smaller, than the errors in either direction.. */
      (environment.get_last_measured_value() <= error_positive_direction)
      &&(environment.get_last_measured_value() <= error_negative_direction)
    )dampening_value = service_context.get_zetta(); /* ..decrease the amount to move the current net */
      else dampening_value = double_literal(1.0); /* reducing oscillation at lower error ranges */
    /* collect the fragment, revert weight changes */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      weight_gradients[weight_index] = ( ( error_positive_direction - error_negative_direction ) / (weight_epsilon) );
      if(save_to_fragment)add_to_fragment( weight_index, (weight_gradients[weight_index] * direction[weight_index] * dampening_value) );
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - (direction[weight_index])) );
    }
    weight_updater->update_solution_with_weights(*net_solution);
  }else throw std::runtime_error("Incompatible direction given to approximate for!");
}

sdouble32 Sparse_net_approximizer::get_single_weight_gradient(uint32 weight_index){
  sdouble32 gradient;
  sdouble32 dampening_value;
  const sdouble32 current_epsilon = service_context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  /* Push the choosen weight in one direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);
  environment.push_state(); /* Approximate the modified weights gradient */
  gradient = -environment.stochastic_evaluation(*solver, iteration);
  environment.pop_state();

  /* Push the selected weight in other direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);
  environment.push_state(); /* Approximate the newly modified weights gradient */
  sdouble32 new_error_state = -environment.stochastic_evaluation(*solver, iteration);
  environment.pop_state();

  /* Calculate the gradient */
  if( /* In case the initial error is smaller, than the errors in either direction.. */
    (environment.get_last_measured_value() <= new_error_state)
    &&(environment.get_last_measured_value() <= gradient)
  )dampening_value = service_context.get_zetta(); /* ..decrease the amount to move the current net */
    else dampening_value = double_literal(1.0); /* reducing oscillation at lower error ranges */
  gradient = -(gradient - new_error_state) * (current_epsilon_double);

  /* Revert weight modification and the error state with it */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);
  return (gradient * dampening_value);
}

sdouble32 Sparse_net_approximizer::get_gradient_for_all_weights(void){
  sdouble32 gradient;
  sdouble32 dampening_value;
  sdouble32 error_negative_direction;
  sdouble32 error_positive_direction;
  const sdouble32 current_epsilon = service_context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  } /* Push every weight in a positive epsilon direction */
  weight_updater->update_solution_with_weights(*net_solution);
  environment.push_state(); /* Approximate the modified weights gradient */
  error_positive_direction = -environment.stochastic_evaluation(*solver, iteration);
  environment.pop_state();

  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  } /* Push the weights to the other direction */
  weight_updater->update_solution_with_weights(*net_solution);
  environment.push_state(); /* Approximate the newly modified weights gradient */
  error_negative_direction = -environment.stochastic_evaluation(*solver, iteration);
  environment.pop_state();

  if( /* In case the initial error is smaller, than the errors in either direction.. */
    (environment.get_last_measured_value() <= error_positive_direction)
    &&(environment.get_last_measured_value() <= error_negative_direction)
  )dampening_value = service_context.get_zetta(); /* ..decrease the amount to move the current net */
    else dampening_value = double_literal(1.0); /* reducing oscillation at lower error ranges */

  gradient = -(error_positive_direction - error_negative_direction) * (current_epsilon_double);
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  } /* Revert weight modifications and the error state with it */
  weight_updater->update_solution_with_weights(*net_solution);
  return (gradient * dampening_value);
}

void Sparse_net_approximizer::add_to_fragment(uint32 weight_index, sdouble32 gradient_fragment_value){
  uint32 values_index = 0;
  uint32 values_index_target = gradient_fragment.values_size();
  uint32 weight_synapse_index_target = gradient_fragment.weight_synapses_size();
  Index_synapse_interval tmp_synapse_interval;

  for(uint32 weight_syn_index = 0; static_cast<sint32>(weight_syn_index) < gradient_fragment.weight_synapses_size(); ++weight_syn_index){
    if( /* If the weight synapse is at or in-between the first index before the start of the synapse.. */
      (
        ((0 < gradient_fragment.weight_synapses(weight_syn_index).starts())
        &&((gradient_fragment.weight_synapses(weight_syn_index).starts()-1) <= static_cast<sint32>(weight_index)))
        ||((0 == gradient_fragment.weight_synapses(weight_syn_index).starts())&&(0 <= weight_index))
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
    ||(static_cast<sint32>(weight_synapse_index_target) >= gradient_fragment.weight_synapses_size())
    ||(static_cast<sint32>(values_index_target) >= gradient_fragment.values_size())
  ){
    gradient_fragment.add_values(gradient_fragment_value);
    tmp_synapse_interval.set_interval_size(1);
    tmp_synapse_interval.set_starts(weight_index);
    *gradient_fragment.add_weight_synapses() = tmp_synapse_interval;
  }else{
    const uint32 synapse_starts = gradient_fragment.weight_synapses(weight_synapse_index_target).starts();
    const uint32 synapse_size = gradient_fragment.weight_synapses(weight_synapse_index_target).interval_size();
    const uint32 synapse_ends = synapse_starts + synapse_size;

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

void Sparse_net_approximizer::apply_fragment(void){
  uint32 fragment_value_index = 0;
  std::fill(applied_direction.begin(),applied_direction.end(), double_literal(0.0));

  if(weight_updater->is_finished())weight_updater->start();
  if(1 == gradient_fragment.weight_synapses_size()){
    std::copy(
      gradient_fragment.values().begin(),
      gradient_fragment.values().begin() + gradient_fragment.weight_synapses(0).interval_size(),
      applied_direction.begin() + gradient_fragment.weight_synapses(0).starts()
    );
  }else{
    Synapse_iterator<>::iterate(gradient_fragment.weight_synapses(), [&](
      Index_synapse_interval interval, sint32 weight_index
    ){
      applied_direction[weight_index] += gradient_fragment.values(fragment_value_index);
      ++fragment_value_index;
    });
  }

  weight_updater->iterate(applied_direction, *net_solution);
  gradient_fragment = Gradient_fragment();
  environment.full_evaluation(*solver);
}

} /* namespace rafko_gym */
