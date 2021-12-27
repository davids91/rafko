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

#include "rafko_net/services/solution_builder.h"

#include <math.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/neuron_router.h"
#include "rafko_net/services/synapse_iterator.h"

#include "rafko_net/services/partial_solution_builder.h"

namespace rafko_net{

Solution* SolutionBuilder::build(const RafkoNet& net, bool optimize_to_gpu){
  NeuronRouter neuron_router(net);
  Solution* solution = google::protobuf::Arena::CreateMessage<Solution>(settings.get_arena_ptr());
  uint32 overall_partial_solution_count = 0u;
  sdouble32 remaining_megabytes_in_row = 0;
  sdouble32 current_neuron_megabyte_size;
  uint32 reach_back_max = 0u;
  uint32 reach_index_max = 0u;
  uint32 current_neuron_index;
  bool has_neuron = false;

  if(0 == net.output_neuron_number()) throw std::runtime_error("Can't build a solution with 0 output Neurons!");
  while(!neuron_router.finished()){ /* Until the whole network is processed */
    if( (!optimize_to_gpu)&&(0 == solution->cols_size()) )
      neuron_router.collect_subset(settings.get_max_solve_threads(),settings.get_device_max_megabytes(), false);
    else neuron_router.collect_subset(settings.get_max_solve_threads(),settings.get_device_max_megabytes(), true);

    remaining_megabytes_in_row = settings.get_device_max_megabytes();
    const sdouble32 max_megabytes_in_one_partial = ( remaining_megabytes_in_row / static_cast<sdouble32>(settings.get_max_solve_threads()) );
    overall_partial_solution_count = solution->partial_solutions_size();

    if(0u < neuron_router.get_subset_size()){
      for(uint32 partial_index_in_row = 0; partial_index_in_row < settings.get_max_solve_threads(); ++partial_index_in_row){
        if(nullptr == settings.get_arena_ptr() ) *solution->add_partial_solutions() = PartialSolution();
        else *solution->add_partial_solutions() = *google::protobuf::Arena::CreateMessage<PartialSolution>(settings.get_arena_ptr());

        /* fill up the partial with Neurons */
        PartialSolution& this_partial = *solution->mutable_partial_solutions(solution->partial_solutions_size()-1);
        sdouble32 remaining_megabytes_in_partial = max_megabytes_in_one_partial;
        has_neuron = neuron_router.get_first_neuron_index_from_subset(current_neuron_index);
        if(has_neuron) current_neuron_megabyte_size = NeuronInfo::get_neuron_estimated_size_megabytes(net.neuron_array(current_neuron_index));
          else break;
        while( /* there are Neurons left to put into this partial */
          ((has_neuron)||(neuron_router.get_first_neuron_index_from_subset(current_neuron_index)))
          &&( /* And the current Neuron is continuing the current partial */
            (0 == this_partial.output_data().interval_size())
            ||(current_neuron_index == (get_last_neuron_index_of_partial(this_partial) + 1u))
          )
        ){
          if(!has_neuron){ /* A new Neuron index was acquired from the router, refresh size info */
            current_neuron_megabyte_size = NeuronInfo::get_neuron_estimated_size_megabytes(net.neuron_array(current_neuron_index));
          }
          if(current_neuron_megabyte_size >= remaining_megabytes_in_partial){
            break;
          }
          if(0u == this_partial.output_data().interval_size()) /* The first Neuron inside the partial solution shall determine its start */
            this_partial.mutable_output_data()->set_starts(current_neuron_index);
          std::pair<uint32,uint32> neuron_input_params = PartialSolutionBuilder::add_neuron_to_partial_solution(net, current_neuron_index, this_partial);
          remaining_megabytes_in_row -= current_neuron_megabyte_size;
          remaining_megabytes_in_partial -= current_neuron_megabyte_size;
          if(reach_back_max < std::get<0>(neuron_input_params))
            reach_back_max = std::get<0>(neuron_input_params);
          if(reach_index_max < std::get<1>(neuron_input_params))
            reach_index_max = std::get<1>(neuron_input_params);
          std::vector<std::reference_wrapper<const FeatureGroup>> features_solved_by_neuron = neuron_router.confirm_first_subset_element_processed(current_neuron_index);
          for(const FeatureGroup& feature : features_solved_by_neuron){ *this_partial.add_solved_features() = feature; }
          has_neuron = neuron_router.get_first_neuron_index_from_subset(current_neuron_index);
        }/* while(able to put Neurons into the current subset) */
        if(0u == this_partial.output_data().interval_size()){
          solution->mutable_partial_solutions()->RemoveLast();
        }
        if( /* in case there are no more available Neurons in the subset */
          (0u == neuron_router.get_subset_size()) /* Or the first partial of the first row is finished.. */
          ||((!optimize_to_gpu)&&(0u == solution->cols_size())) /* ..while optimizing solution to CPU */
        )break;
        /*!Note: The first partial of the first row collected Neurons in a non-strict way,
         * so other Neurons might not fit into other partials, because they might have dependencies in this row
         */
      }

      if(solution->partial_solutions_size() > static_cast<sint32>(overall_partial_solution_count))
        solution->add_cols(solution->partial_solutions_size() - overall_partial_solution_count);
      neuron_router.reset_remaining_subset(); /* Whichever Neuron coudn't fit into the partial shall have its state reset */
    } /* if(0u < neuron_router.get_subset_size()) */
  } /* while(!neuron_router.finished()) */

  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_back_max + 1u); /* Current loop is "0" reachback, so length should be at least 1 */
  solution->set_network_input_size(reach_index_max + 1u);
  assert( net.input_data_size() == reach_index_max + 1u);
  return solution;
}

} /* namespace rafko_net */
