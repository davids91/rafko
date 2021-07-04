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
#include <deque>
#include <memory>
#include <stdexcept>

#include "sparse_net_library/models/neuron_info.h"
#include "sparse_net_library/services/neuron_router.h"
#include "sparse_net_library/services/synapse_iterator.h"

#include "sparse_net_library/services/partial_solution_builder.h"

namespace sparse_net_library{

using std::deque;
using std::unique_ptr;
using std::ref;
using std::lock_guard;

Solution* Solution_builder::build(const SparseNet& net){
  if(0 == net.output_neuron_number()) throw std::runtime_error("Can't build a solution with 0 output Neurons!");
  Neuron_router neuron_router(net);
  Solution* solution = google::protobuf::Arena::CreateMessage<Solution>(service_context.get_arena_ptr());
  uint32 partial_solution_starting_index_in_this_row = 0;
  uint32 overall_partial_solution_count = 0;
  uint32 remaining_bytes_in_row = 0;
  uint32 reach_back; /* for each partial solution in the currently built row */
  uint32 reach_back_max = 0;
  uint32 current_neuron_byte_size;
  uint32 current_neuron_index;
  bool has_neuron;

  while(!neuron_router.finished()){ /* Until the whole network is processed */
    if(0 < remaining_bytes_in_row){ /* If there are bytes left in the current row to spare */
      /* Collect a strict subset, to extend the current partial solutions */
      neuron_router.collect_subset(service_context.get_max_solve_threads(),service_context.get_device_max_megabytes(), true);
      while( /* try to fit the subset into the previous row of partial solutions */
        (0 < remaining_bytes_in_row)&&(0 < neuron_router.get_subset_size())
        &&(0 < solution->cols_size()) /* ..previous rows are required for that... */
      ){
        /* Collect the first Neuron from the subset */
        has_neuron = neuron_router.get_first_neuron_index_from_subset(current_neuron_index);
        if(!has_neuron) break;

        /* Try to extend the partials until free bytes remain in the device */
        current_neuron_byte_size = Neuron_info::get_neuron_estimated_size_bytes(net.neuron_array(current_neuron_index));
        uint32 current_row_size = solution->cols(solution->cols_size()-1);
        for(uint32 current_col_iterator = 0; current_col_iterator < current_row_size; ++current_col_iterator){
          uint32 current_col_index = partial_solution_starting_index_in_this_row + current_col_iterator;
          Partial_solution& partial = *solution->mutable_partial_solutions(current_col_index);
          while( /* able to collect a Neuron from the subset which fits in it */
            ((has_neuron)||(neuron_router.get_first_neuron_index_from_subset(current_neuron_index)))
            &&(current_neuron_byte_size < remaining_bytes_in_row) /* there are bytes left to fit the Neuron */
            &&( /* the neuron is continuing the last neuron in the current partial */
              (0 == partial.output_data().interval_size()) /* Although this line should never be true... */
              ||(current_neuron_index == (get_last_neuron_index_of_partial(partial) + 1u))
            )
          ){
            if(!has_neuron){ /* A new Neuron index was acquired from the router, refresh size info */
              current_neuron_byte_size = Neuron_info::get_neuron_estimated_size_bytes(net.neuron_array(current_neuron_index));
            }
            /* Add the Neuron into the partial solution */
            reach_back = Partial_solution_builder::add_neuron_to_partial_solution(net, current_neuron_index, partial);
            if(reach_back_max < reach_back)reach_back_max = reach_back;
            remaining_bytes_in_row -= current_neuron_byte_size;
            if(neuron_router.confirm_first_subset_element_processed(current_neuron_index)){
              has_neuron = neuron_router.get_first_neuron_index_from_subset(current_neuron_index);
              if(has_neuron){ /* A new Neuron index was acquired from the router, refresh size info */
                current_neuron_byte_size = Neuron_info::get_neuron_estimated_size_bytes(net.neuron_array(current_neuron_index));
              }
            }
          } /* while(able to fit a new Neuron into the current partial solution) */
        }/* for(every partial solution in the latest row) */
      }
    }else{
      neuron_router.collect_subset(service_context.get_max_solve_threads(),service_context.get_device_max_megabytes(), false);
    }

    /* try to generate a row to fit into `max threads` partial solutions */
    remaining_bytes_in_row = neuron_router.get_subset_size_bytes();
    uint32 max_neurons_in_one_partial = (neuron_router.get_subset_size() / service_context.get_max_solve_threads());
    uint32 partial_number_max = service_context.get_max_solve_threads();
    overall_partial_solution_count = solution->partial_solutions_size();
    partial_solution_starting_index_in_this_row = overall_partial_solution_count;

    /* Generate appropriate number of partial solutions and put the neurons inside */
    /* !Note It is supposed that the subset is sorted in ascending order */
    for(uint32 partial_index_in_row = 0; partial_index_in_row < partial_number_max; ++partial_index_in_row){
      if(nullptr == service_context.get_arena_ptr() ){
        *solution->add_partial_solutions() = Partial_solution();
      }else{
        *solution->add_partial_solutions() = *google::protobuf::Arena::CreateMessage<Partial_solution>(service_context.get_arena_ptr());
      }

      /* fill up the partial with Neurons */
      Partial_solution& this_partial = *solution->mutable_partial_solutions(solution->partial_solutions_size()-1);
      uint32 neurons_in_this_partial = 0;
      reach_back_max = 0;
      while( /* there are Neurons left to put into this partial */
        (neurons_in_this_partial < max_neurons_in_one_partial)
        &&((has_neuron)||(neuron_router.get_first_neuron_index_from_subset(current_neuron_index)))
        &&( /* And the current Neuron is continuing the current partial */
          (0 == this_partial.output_data().interval_size())
          ||(current_neuron_index == (get_last_neuron_index_of_partial(this_partial) + 1u))
        )
      ){
        if(!has_neuron){ /* A new Neuron index was acquired from the router, refresh size info */
          current_neuron_byte_size = Neuron_info::get_neuron_estimated_size_bytes(net.neuron_array(current_neuron_index));
        }
        reach_back = Partial_solution_builder::add_neuron_to_partial_solution(net, current_neuron_index, this_partial);
        if(reach_back_max < reach_back)reach_back_max = reach_back;
        if(neuron_router.confirm_first_subset_element_processed(current_neuron_index)){
          has_neuron = neuron_router.get_first_neuron_index_from_subset(current_neuron_index);
        }
      }/* while(able to put Neurons into the current subset) */

      if(0 == this_partial.output_data().interval_size()){ /* Unable to put a Neuron into this partial */
        solution->mutable_partial_solutions()->RemoveLast();
      } /*!Note: this shouldn't happen */
    }
    solution->add_cols(solution->partial_solutions_size() - overall_partial_solution_count);

    /* Whichever Neuron coudn't fit into the partial shall have its state reset */
    remaining_bytes_in_row = neuron_router.get_subset_size_bytes();
    neuron_router.reset_remaining_subset();

  } /* while(!neuron_router.finished()) */

  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_back_max + 1); /* Current loop is "0" reachback, so length should be at least 1 */
  return solution;
}

} /* namespace sparse_net_library */
