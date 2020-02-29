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

#include "services/solution_builder.h"

#include "models/neuron_info.h"
#include "services/neuron_router.h"
#include "services/partial_solution_builder.h"

#include <memory>

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
  uint32 neuron_index;
  uint32 row_iterator = 0;
  Neuron_router net_iterator(net);
  uint32 placed_neurons_in_partial = 0;
  uint32 placed_neurons_in_row = 0;
  uint32 partial_output_synapse_count = 0;
  uint32 latest_placed_neuron_index = net.neuron_array_size();
  bool strict_mode = false;

  if(0 == net.output_neuron_number()) throw "Can't build a solution with 0 output Neurons!";
  while(!net_iterator.finished()){ /* Until the whole output layer is processed */
    net_iterator.collect_subset(arg_max_solve_threads,arg_device_max_megabytes, strict_mode); /* Collect solvable neuron indices */
    placed_neurons_in_partial = net_iterator.get_subset_size();
    while (
      ((current_partial->SpaceUsedLong() /* Bytes */ / 1024.0 /* KB *// 1024.0 /* MB */) <= arg_device_max_megabytes)
      &&(0 < placed_neurons_in_partial)
    ){
      placed_neurons_in_partial = 0;
      while( /* Put all collected Neurons into the current @Partial_solution */
        ((current_partial->SpaceUsedLong() /* Bytes */ / 1024.0 /* KB *// 1024.0 /* MB */) <= arg_device_max_megabytes)
        &&(placed_neurons_in_row < net_iterator.get_subset_size())
      ){
        neuron_index = net_iterator[placed_neurons_in_row];
        partial_builder->add_neuron_to_partial_solution(neuron_index);
        ++placed_neurons_in_row;
        neurons_in_row.push_back(neuron_index);
        if(
          (0 < partial_output_synapse_count)
          &&(0 < neuron_index)
          &&(neuron_index-1) != latest_placed_neuron_index
        ){ /* If last placed Neuron is not the one at the previous index */
          partial_output_synapse_count = 0;
        }
        latest_placed_neuron_index = neuron_index;
        Partial_solution_builder::add_to_synapse( /* Neural input shall be added from the input of the @Partial_solution */
          neuron_index, partial_output_synapse_count,
          current_partial->mutable_output_data()
        );
      }
    } /* Loop for placing the Neurons from the subset into the Partial Solutions */
    if( /* If no Neurons could be placed inside the partial_matrix, a new row is needed */
      (current_partial->SpaceUsedLong() /* Bytes *// 1024.0 /* KB *// 1024.0 /* MB */) < arg_device_max_megabytes
      &&(0 == placed_neurons_in_partial)
    ){
      if(0 == partial_matrix.back().back()->internal_neuron_number()){
        partial_matrix.back().pop_back(); /* Remove the last column, since it's empty */
      }else current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);

      if(0 == partial_matrix.back().size()) partial_matrix.pop_back(); /* The last @Partial_solution row has zero elements */
      partial_matrix.push_back(vector<Partial_solution*>(1,current_partial));
      partial_builder.reset();
      partial_builder = std::make_unique<Partial_solution_builder>(net, *current_partial);
      strict_mode = false;
      for(uint32 neuron_index_in_row : neurons_in_row){
        net_iterator.confirm_first_subset_element_processed(neuron_index_in_row);
      }
      neurons_in_row.clear();
      placed_neurons_in_row = 0;
      ++row_iterator;
    }else if((current_partial->SpaceUsedLong() /* Bytes *// 1024.0 /* KB *// 1024.0 /* MB */) >= arg_device_max_megabytes){
      /* Put a new Partial_solution into the current row if the memory limit is reached */
      current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
      partial_matrix[row_iterator].push_back(current_partial); /* In case the @Partial_solution reached the size limit, push in a new one */
      partial_builder.reset();
      partial_builder = std::make_unique<Partial_solution_builder>(net, *current_partial);
      net_iterator.reset_remaining_subset();
      strict_mode = true;
    }
  } /* while(!net_iterator.finished()) */
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
  for(vector<Partial_solution*> row : partial_matrix){
    solution->add_cols(row.size());
    for(Partial_solution* cell : row){
      *solution->add_partial_solutions() = *cell;
    }
  } /* Build the @Solution from the @Partial_Solution matrix */
  return solution;
}

} /* namespace sparse_net_library */
