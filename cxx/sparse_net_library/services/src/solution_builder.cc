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

#include <memory>
#include <stdexcept>

#include "sparse_net_library/models/neuron_info.h"

#include "sparse_net_library/services/neuron_router.h"
#include "sparse_net_library/services/partial_solution_builder.h"

namespace sparse_net_library{

Solution* Solution_builder::build(const SparseNet& net ){

  using std::ref;
  using std::unique_ptr;

  Partial_solution* current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
  Solution* solution = google::protobuf::Arena::CreateMessage<Solution>(arg_arena_ptr);
  vector<vector<Partial_solution*>> partial_matrix = vector<vector<Partial_solution*>>(
    1, vector<Partial_solution*>(1,current_partial)
  );
  unique_ptr<Partial_solution_builder> partial_builder =
    std::make_unique<Partial_solution_builder>(net, *current_partial);
  vector<uint32> neurons_in_row = vector<uint32>();
  uint32 row_iterator = 0;
  Neuron_router neuron_router(net);
  uint32 placed_neurons_in_partial = 0;
  uint32 placed_neurons_in_row = 0;
  uint32 partial_output_synapse_count = 0;
  uint32 latest_placed_neuron_index = net.neuron_array_size();
  uint32 reach_back;
  bool strict_mode = false;

  if(0 == net.output_neuron_number()) throw std::runtime_error("Can't build a solution with 0 output Neurons!");
  while(!neuron_router.finished()){ /* Until the whole output layer is processed */
    neuron_router.collect_subset(arg_max_solve_threads,arg_device_max_megabytes, strict_mode); /* Collect solvable neuron indices */
    placed_neurons_in_partial = 1; /* To enter the placement loop.. */
    while (
      ((current_partial->SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */) <= arg_device_max_megabytes)
      &&(0 < placed_neurons_in_partial)
    ){ /* no Neurons were placed in the current loop and there's still some space left: let's try to place some Neurons! */
      placed_neurons_in_partial = 0;
      while( /* While there is space and.. */
        ((current_partial->SpaceUsedLong() /* Bytes */ / double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */) <= arg_device_max_megabytes)
        &&(placed_neurons_in_row < neuron_router.get_subset_size()) /* ..There are Neurons from the subset to place */
      ){ /* Put as many collected Neurons into the current @Partial_solution as feasible */
        reach_back = partial_builder->add_neuron_to_partial_solution(neuron_router[placed_neurons_in_row]);
        if(reach_past_loops_maximum < reach_back)
          reach_past_loops_maximum = reach_back;
        neurons_in_row.push_back(neuron_router[placed_neurons_in_row]);
        if(
          (0 < partial_output_synapse_count)
          &&(0 < neuron_router[placed_neurons_in_row])
          &&(neuron_router[placed_neurons_in_row]-1) != latest_placed_neuron_index
        ){ /* If last placed Neuron is not the one at the previous index */
          partial_output_synapse_count = 0;
        }
        latest_placed_neuron_index = neuron_router[placed_neurons_in_row];
        Partial_solution_builder::add_to_synapse( /* Neural input shall be added from the input of the @Partial_solution */
          neuron_router[placed_neurons_in_row], partial_output_synapse_count,
          current_partial->mutable_output_data()
        );
        ++placed_neurons_in_row;
        ++placed_neurons_in_partial;
      }
    } /* Loop for placing the Neurons from the subset into the Partial Solutions */
    if( /* If no Neurons could be placed inside the partial_matrix, or there's no space left */
      (current_partial->SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */) > arg_device_max_megabytes
      ||(0 == placed_neurons_in_partial)
    ){ /* A new row shall be added into the @Solution */
      for(uint32 neuron_index_in_row : neurons_in_row){
        neuron_router.confirm_first_subset_element_processed(neuron_index_in_row);
      }
      neurons_in_row.clear();
      neuron_router.reset_remaining_subset();
      placed_neurons_in_row = 0;

      if(0 == partial_matrix.back().back()->internal_neuron_number()){
        partial_matrix.back().pop_back(); /* Remove the last column, since it's empty */
      }else current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
      if(0 == partial_matrix.back().size()) partial_matrix.pop_back(); /* The last @Partial_solution row has zero elements */
      partial_matrix.push_back(vector<Partial_solution*>(1,current_partial));
      partial_builder.reset();
      partial_builder = std::make_unique<Partial_solution_builder>(net, *current_partial);
      placed_neurons_in_partial = 0;
      strict_mode = false;
      ++row_iterator;
    }else if( /* If there is space left and there are still Neurons left to place in the subset */
      ((current_partial->SpaceUsedLong() /* Bytes *// double_literal(1024.0) /* KB *// double_literal(1024.0) /* MB */) < arg_device_max_megabytes)
      &&(placed_neurons_in_row < neuron_router.get_subset_size())
    ){
      /* Put a new Partial_solution into the current row if the memory limit is reached */
      /*!Note: A new partial shall never reset the subset, as it the contents of the subset store which 
       * Neurons can be solved in paralell. Resetting this would make the Builder disregard transitive dependencies.
       * */
      neuron_router.reset_all_except(neurons_in_row);
      current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
      partial_matrix[row_iterator].push_back(current_partial); /* In case the @Partial_solution reached the size limit, push in a new one */
      partial_builder.reset();
      partial_builder = std::make_unique<Partial_solution_builder>(net, *current_partial);
      placed_neurons_in_partial = 0;
      strict_mode = true;
    }
  } /* while(!neuron_router.finished()) */
  if(0 == partial_matrix.back().back()->internal_neuron_number()) /* The last @Partial_solution has zero Neurons in it */
  {
    partial_matrix.back().pop_back();
  }

  if(0 == partial_matrix.back().size()) /* The last @Partial_solution row has zero elements */
  {
    partial_matrix.pop_back();
  }

  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_past_loops_maximum + 1); /* Current loop is "0" reachback, so length should be at least 1 */
  for(vector<Partial_solution*> row : partial_matrix){
    solution->add_cols(row.size());
    for(Partial_solution* cell : row){
      *solution->add_partial_solutions() = *cell;
    }
  } /* Build the @Solution from the @Partial_Solution matrix */
  return solution;
}

} /* namespace sparse_net_library */
