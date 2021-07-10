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
#include "sparse_net_library/services/synapse_iterator.h"

namespace rafko_gym{

using std::make_unique;
using std::lock_guard;
using std::ref;

using sparse_net_library::Synapse_iterator;
using sparse_net_library::Index_synapse_interval;

void Sparse_net_approximizer::collect_approximates_from_weight_gradients(void){
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  uint32 index_of_biggest = 0;
  sdouble32 sum_gradient = double_literal(0.0);
  sdouble32 gradient_overview = get_gradient_for_all_weights() * context.get_step_size();

  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] += get_single_weight_gradient(weight_index);
    sum_gradient += std::pow(weight_gradients[weight_index],double_literal(2.0));
    if(std::abs(weight_gradients[index_of_biggest]) < std::abs(weight_gradients[weight_index]))
      index_of_biggest = weight_index;
  }
  sum_gradient = std::sqrt(sum_gradient);

  convert_direction_to_gradient(last_applied_direction,false); /* check the last applied direction */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = (
      ( /* Gradients normalized by the biggest value */
        (weight_gradients[weight_index] / std::abs(weight_gradients[index_of_biggest]))
        + gradient_overview /* plus the overview gradient */
      ) / (double_literal(1.0) + std::abs(gradient_overview)) /* normalized by the extended value */
    ); /*!Note: the biggest value in the weight gradients should be at most 1.0 after normalization,
        * so dividing by 1.0 + gradient_overview should normalize the offseted gradients
        */
    weight_gradients[weight_index] *= context.get_step_size();
  }

  convert_direction_to_gradient(weight_gradients,true);
}

void Sparse_net_approximizer::collect_approximates_from_random_direction(void){
  vector<sdouble32> direction(net.weight_table_size());
  sdouble32 gradient_overview = get_gradient_for_all_weights() * context.get_step_size();
  std::generate_n(direction.begin(),net.weight_table_size(),[=](){
    return(
      (
        (static_cast<sdouble32>(rand()%201) / double_literal(200.0)) - double_literal(1.0) + gradient_overview
      ) * context.get_step_size()
    );
  });
  convert_direction_to_gradient(direction,true);
}

void Sparse_net_approximizer::convert_direction_to_gradient(vector<sdouble32>& direction, bool save_to_fragment){
  if(net.weight_table_size() == static_cast<sint32>(direction.size())){
    sdouble32 dampening_value;
    sdouble32 error_negative_direction;
    sdouble32 error_positive_direction;
    vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
    vector<sdouble32> weight_steps(net.weight_table_size(),double_literal(0.0));
    sdouble32 weight_epsilon = double_literal(0.0);

    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      weight_steps[weight_index] = direction[weight_index];
      weight_epsilon += std::pow(weight_steps[weight_index], double_literal(2.0));
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - weight_steps[weight_index]) );
    } /* apply the direction on which network approximation shall be done */
    weight_updater->update_solution_with_weights(*net_solution);
    weight_epsilon = std::sqrt(weight_epsilon) * double_literal(2.0);

    /* see the error values at the negative end of the current direction */
    environment.push_state();
    error_negative_direction = environment.stochastic_evaluation(*solver);

    /* see the error values at the positive end of the current direction */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index)
      net.set_weight_table(weight_index, (net.weight_table(weight_index) + (weight_steps[weight_index] * double_literal(2.0))) );
    weight_updater->update_solution_with_weights(*net_solution);
    error_positive_direction = environment.stochastic_evaluation(*solver);
    environment.pop_state(); /* Restore train set to previous error state, decide if dampening is needed */
    if( /* In case the initial error is smaller, than the errors in either direction.. */
      // (train_set.get_error_avg() < error_positive_direction)
      // &&(train_set.get_error_avg() < error_negative_direction)
      true /* It was always true... */
    )dampening_value = context.get_zetta(); /* ..decrease the amount to move the current net */
      else dampening_value = double_literal(1.0); /* reducing oscillation at lower error ranges */

    /* collect the fragment, revert weight changes */
    for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
      weight_gradients[weight_index] = (
        ( error_positive_direction - error_negative_direction ) / (weight_epsilon)
      );

      if(save_to_fragment)add_to_fragment( weight_index, (weight_gradients[weight_index] * weight_steps[weight_index] * dampening_value) );
      net.set_weight_table(weight_index, (net.weight_table(weight_index) - (weight_steps[weight_index])) );
    }
    weight_updater->update_solution_with_weights(*net_solution);
  }else throw std::runtime_error("Incompatible direction given to apprximate for!");
}

void Sparse_net_approximizer::collect_fragment(void){
  /* Approximate the gradient for every weight */
  vector<sdouble32> weight_gradients(net.weight_table_size(),double_literal(0.0));
  uint32 index_of_biggest = double_literal(0.0);
  sdouble32 average_gradient = double_literal(0.0);
  sdouble32 sum_gradient = double_literal(0.0);
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    weight_gradients[weight_index] = get_single_weight_gradient(weight_index);
    average_gradient += weight_gradients[weight_index];
    sum_gradient += std::pow(weight_gradients[weight_index],double_literal(2.0));
    if(weight_gradients[index_of_biggest] < weight_gradients[weight_index])
      index_of_biggest = weight_index;
  }
  sum_gradient = std::sqrt(sum_gradient);
  average_gradient /= static_cast<sdouble32>(net.weight_table_size());
  add_to_fragment(index_of_biggest, (weight_gradients[index_of_biggest] / sum_gradient) );
}

sdouble32 Sparse_net_approximizer::get_single_weight_gradient(uint32 weight_index){
  sdouble32 gradient;
  const sdouble32 current_epsilon = context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  /* Save error from the initial network, decide the minibatch and sequence truncation */
  environment.push_state();

  /* Push it in one direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);

  /* Approximate the modified weights gradient */
  gradient = environment.stochastic_evaluation(*solver);

  /* Push the selected weight in other direction */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);

  /* Approximate the newly modified weights gradient */
  sdouble32 new_error_state = environment.stochastic_evaluation(*solver);
  gradient = -(gradient - new_error_state) / (current_epsilon_double * context.get_minibatch_size());

  /* Revert weight modification and the error state with it */
  net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  weight_updater->update_solution_with_weight(*net_solution, weight_index);
  environment.pop_state();

  return gradient;
}

sdouble32 Sparse_net_approximizer::get_gradient_for_all_weights(void){
  sdouble32 gradient;
  sdouble32 error_negative_direction;
  sdouble32 error_positive_direction;
  const sdouble32 current_epsilon = context.get_sqrt_epsilon();
  const sdouble32 current_epsilon_double = current_epsilon * double_literal(2.0);

  environment.push_state(); /* Save error from the initial network, decide the minibatch and sequence truncation */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  } /* Push every weight in a positive epsilon direction */
  weight_updater->update_solution_with_weights(*net_solution);
  error_positive_direction = environment.stochastic_evaluation(*solver); /* Approximate the modified weights gradient */
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) - current_epsilon_double) );
  } /* Push the weights to the other direction */
  weight_updater->update_solution_with_weights(*net_solution);
  error_negative_direction = environment.stochastic_evaluation(*solver); /* Approximate the newly modified weights gradient */
  gradient = -(error_positive_direction - error_negative_direction) / (current_epsilon_double * context.get_minibatch_size());
  for(uint32 weight_index = 0; static_cast<sint32>(weight_index) < net.weight_table_size(); ++weight_index){
    net.set_weight_table(weight_index, (net.weight_table(weight_index) + current_epsilon) );
  } /* Revert weight modifications and the error state with it */
  weight_updater->update_solution_with_weights(*net_solution);
  environment.pop_state();
  return gradient;
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
      insert_element_at_position(
        *gradient_fragment.mutable_values(), gradient_fragment_value, (values_index_target + synapse_size)
      );
    }
  }
}

void Sparse_net_approximizer::apply_fragment(void){
  uint32 fragment_value_index = 0;
  std::fill(last_applied_direction.begin(),last_applied_direction.end(),double_literal(0.0));

  if(weight_updater->is_finished())weight_updater->start();
  if(1 == gradient_fragment.weight_synapses_size()){
    std::copy(
      gradient_fragment.values().begin(),
      gradient_fragment.values().begin() + gradient_fragment.weight_synapses(0).interval_size(),
      last_applied_direction.begin() + gradient_fragment.weight_synapses(0).starts()
    );
  }else{
    Synapse_iterator<>::iterate(gradient_fragment.weight_synapses(), [&](
      Index_synapse_interval interval, sint32 weight_index
    ){
      last_applied_direction[weight_index] += gradient_fragment.values(fragment_value_index);
      ++fragment_value_index;
    });
  }

  weight_updater->iterate(last_applied_direction, *net_solution);
  gradient_fragment = Gradient_fragment();
}

} /* namespace rafko_gym */
