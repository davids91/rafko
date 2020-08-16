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

#include "sparse_net_library/services/solution_builder.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "sparse_net_library/models/neuron_info.h"

#include "sparse_net_library/services/partial_solution_builder.h"

namespace sparse_net_library{

using std::unique_ptr;
using std::ref;
using std::lock_guard;

Solution* Solution_builder::build(const SparseNet& net ){

  Neuron_router neuron_router(net);
  Solution* solution = google::protobuf::Arena::CreateMessage<Solution>(arg_arena_ptr);
  deque<uint32> partial_indices_in_row;
  vector<uint32> last_index_in_partial; /* Contains the last valid index to be included into the partial solution */
  /*!Note: This also implicitly implies that the size of this array also represents the number of partial solutions
   * to be built in the current row. */
  uint32 reach_back;
  uint32 reach_back_max = 0;
  uint32 partial_index_in_row;
  uint32 partial_index_in_solution = 0;
  uint32 first_neuron_in_partial;
  uint32 neuron_count_in_partial;

  if(0 == net.output_neuron_number()) throw std::runtime_error("Can't build a solution with 0 output Neurons!");
  while(!neuron_router.finished()){ /* Until the whole output layer is processed */
    neuron_router.collect_subset(arg_max_solve_threads,arg_device_max_megabytes, true); /* Collect solvable neuron indices */
    /* Decide the place for every Neuron in the current subset by adssigning a partial solution column index to it */
    partial_indices_in_row = deque<uint32>(neuron_router.get_subset_size());
    last_index_in_partial = vector<uint32>();
    for(uint32 subset_index = 0; subset_index < neuron_router.get_subset_size(); ++subset_index){
      /* Try to find the suitable partial of the current subset item */
      for(partial_index_in_row = 0; partial_index_in_row < last_index_in_partial.size(); ++partial_index_in_row){
        if(( /* if the neuron index under subset_index is the next one after the previous top for this partial */
            neuron_router.get_neuron_index_from_subset(subset_index) 
            == (last_index_in_partial[partial_index_in_row] + 1)
          )||(0 == subset_index) /* Or this is the very first neuron in the subset */
        ){ /* assign the current partial index to it */
          break;
        }
      }
      if(partial_index_in_row < last_index_in_partial.size()){ /* If a suitable partial have been found for the subset */
        ++last_index_in_partial[partial_index_in_row]; /* Essentially this means that the new index is taken over */
        partial_indices_in_row[subset_index] = partial_index_in_row; /* assign the partial index to the subset item */
      }else{ /* unable to find a suitable partial, add a new one! */
        partial_indices_in_row[subset_index] = last_index_in_partial.size();
        last_index_in_partial.push_back(subset_index);
      }
    }/* for(subset_index : neuron_router.get_subset_size()) */

    /* Build the partial solutions based on the assigned indices */
    if(0 < last_index_in_partial.size())solution->add_cols(last_index_in_partial.size());
    for(partial_index_in_row = 0; partial_index_in_row < last_index_in_partial.size(); ++partial_index_in_row){ /* for every assigned partial index */

      /* Add a new partial to the solution, and create a builder for it */
      *solution->add_partial_solutions() = *google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
      partial_index_in_solution = solution->partial_solutions_size() - 1; /* It's basically guaranteed by the preceeding line, that (size > 0) */

      /* Scan the subset for Neurons assigned to the current partial index */
      neuron_count_in_partial = 0;
      first_neuron_in_partial = net.neuron_array_size();
      for(uint32 subset_index = 0; subset_index < partial_indices_in_row.size(); ++subset_index){
        if(partial_indices_in_row[subset_index] == partial_index_in_row){ /* if a Neuron in the subset is assigned for the current partial index */
          if(
            ( /* If there is enough space left to put the Neuron into the partial solution */
              solution->partial_solutions(partial_index_in_solution).SpaceUsedLong()
              /* Bytes */ / double_literal(1024.0) /* KB */ / double_literal(1024.0) /* MB */ 
            )
            <= arg_device_max_megabytes
          ){
            reach_back = Partial_solution_builder::add_neuron_to_partial_solution(
              net, neuron_router.get_neuron_index_from_subset(subset_index),
              *solution->mutable_partial_solutions(partial_index_in_solution)
            );
            if(reach_back_max < reach_back)reach_back_max = reach_back;
            partial_indices_in_row[subset_index] = partial_indices_in_row.size() + 1;
            /*!Note: Worst case there will be as many partial solutions as neurons in the current row,
             * so the theoritical maximum of the number of partial solutions in the current wor is the size of the subset.
             * Every Neuron successfully placed shall be marked by this value assignment,
             * so the Neurons left out from partials can be detected.
             **/
            ++neuron_count_in_partial;
            if(net.neuron_array_size() == static_cast<sint32>(first_neuron_in_partial))
              first_neuron_in_partial = neuron_router.get_neuron_index_from_subset(subset_index);
          }/* else no space left in the partial to put the Neuron in! */
        } /* If neuron is placable.. */
      } /* for( subset_index : partial_indices_in_row.size() ) */
      if(
        (0 < neuron_count_in_partial)
        &&(net.neuron_array_size() > static_cast<sint32>(first_neuron_in_partial))
      ){
        *solution->mutable_partial_solutions(partial_index_in_solution)->mutable_output_data() = Index_synapse_interval();
        solution->mutable_partial_solutions(partial_index_in_solution)->mutable_output_data()->set_starts(first_neuron_in_partial);
        solution->mutable_partial_solutions(partial_index_in_solution)->mutable_output_data()->set_interval_size(neuron_count_in_partial);
      }
    } /* for( partial_index : last_index_in_partial.size()) */
    while(0 < neuron_router.get_subset_size()){ /* trying to confirm the first element until the subset is gone */
      if(partial_indices_in_row.size() < partial_indices_in_row[0]){ /* If the Neuron is placed */
        neuron_router.confirm_first_subset_element_processed(neuron_router.get_subset()[0]);
        partial_indices_in_row.pop_front(); /* Also remove the coreresponding partial index from the helper array */
      }else{ /* Neuron could not be placed, omit it */
        neuron_router.confirm_first_subset_element_ommitted(neuron_router.get_subset()[0], partial_indices_in_row);
      }
    }
  } /* while(!neuron_router.finished()) */

  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_back_max + 1); /* Current loop is "0" reachback, so length should be at least 1 */
  return solution;
}

} /* namespace sparse_net_library */
