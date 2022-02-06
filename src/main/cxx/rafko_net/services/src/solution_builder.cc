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

#if(RAFKO_USES_OPENCL)
#include <regex>

#include "rafko_mainframe/models/rafko_settings.h"

#include "rafko_net/models/transfer_function.h"
#include "rafko_net/models/spike_function.h"
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/neuron_router.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_feature_executor.h"

#include "rafko_net/services/partial_solution_builder.h"

namespace rafko_net{

std::unique_ptr<Solution> SolutionBuilder::build(const RafkoNet& net, bool optimize_to_gpu){
  NeuronRouter neuron_router(net);
  std::unique_ptr<Solution> solution = std::make_unique<Solution>();
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

#if(RAFKO_USES_OPENCL)
std::string SolutionBuilder::get_kernel_for_solution(
  const Solution& solution, std::string name, uint32 sequence_size, uint32 prefill_input_num,
  const rafko_mainframe::RafkoSettings& settings
){
  /*!Note: Network solve starts from the first memory/sequence slot of the outputs buffer
   * which is calculated from get_global_id(0) and the size of the max(sequence, neuron_memory);
   * UNLESS mode variable input is non-zero.
   * If the mode varaible is non-zero the start of the execution shall start at the end of the
   * last output slice ( the last neuron value array the kernel is supposed to be updating )
   * and every other output slice before that is considered outputs from the past
   */
  std::string source_base = R"(
    void kernel $$name$$(
       __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
       __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      if( (input_sizes_size == 3) && (output_sizes_size == 1) ){
        const int sequence_index = get_global_id(0);
        const int neuron_array_size = $$neuron_array_size$$;
        const int max_backreach = $$neuron_memory_size$$ - 1;
        const int sequence_size = $$sequence_size$$;
        const int prefill_input_num = $$prefill_input_num$$;
        const int network_input_size = $$network_input_size$$;
        int output_start;
        int current_max_backreach;
        int input_start;
        int label_index = 0; /* index inside the current inside sequence */
        int sequence_start;
        int sequence_max_index;
        if(inputs[0] == 0){ /* normal evaluation */
          current_max_backreach = 0;
          input_start = $$weight_table_offset$$ + (sequence_index * (sequence_size + prefill_input_num) * network_input_size);
          output_start = sequence_index * (sequence_size + prefill_input_num) * neuron_array_size;
          sequence_start = 0;
          sequence_max_index = max(( sequence_size + prefill_input_num ), 1) - 1;
        }else{ /* run-once with memory */
          current_max_backreach = max_backreach;
          input_start = $$weight_table_offset$$;
          output_start = (max_backreach * neuron_array_size); /*$$output_start$$*/
          sequence_start = max(( sequence_size + prefill_input_num ), 1) - 1;
          sequence_max_index = sequence_start;
        }
        for(int label_index = sequence_start; label_index <= sequence_max_index; ++label_index){
          double neuron_partial_result;
          /* +++ GENERATED_NEURON_CODE +++ */
          $$neuron_operations$$
          /* --- GENERATED_NEURON_CODE --- */

          input_start += network_input_size;
          output_start += neuron_array_size;
          if(current_max_backreach < max_backreach)
            ++current_max_backreach;
        }
      }/*if(input sizes match)*/
    }/* kernel */
  )";
  source_base = std::regex_replace(source_base, std::regex("\\$\\$output_start\\$\\$"), std::to_string(
    (solution.network_memory_length()-1) * solution.neuron_number()
  ));
  TransferFunction transfer_function(settings);
  uint32 weight_table_offset = 1u; /* the first input number is the mode of the solution, so weight table starts after that */
  /*!Note: Weight table is supposed to consist of the weight tables of the partial solutions
   * in the order they are inside the @Solution. Inside the kernel this order is used
   * to query the wieghts by position in the device weight table.
   * This note is strictly for one kernel solutions! Networks spanning multiple devices
   * are to re-think this logic.
   */
  std::string neuron_operations;
  std::function<std::string(uint32, std::string)> past_reach_guard = [](uint32 past_reach, std::string content){
    return "( (min(current_max_backreach,max_backreach) < " + std::to_string(past_reach) + " )?(0.0):( " + content + ") )";
  };
  bool feature_locals_declared = false;
  for(const PartialSolution& partial : solution.partial_solutions()){
    SynapseIterator<InputSynapseInterval> partial_input_synapses(partial.input_data());
    uint32 input_synapse_index = 0u;
    uint32 input_synapse_index_offset = 0u;
    uint32 input_start_offset = 0u;
    uint32 weight_synapse_start = 0u;

    for(uint32 inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
      bool first_weight_in_synapse = true;
      uint32 spike_weight_index;
      std::string inner_neuron_operation = "neuron_partial_result = (";
      std::string inner_neuron_input_weight_pairs = "";
      SynapseIterator<>::iterate(partial.weight_indices(),[&](sint32 weight_index){
        if(first_weight_in_synapse){
          first_weight_in_synapse = false;
          spike_weight_index = weight_index;
        }else{
          inner_neuron_input_weight_pairs += std::string("\n");
          if(input_synapse_index_offset < partial.index_synapse_number(inner_neuron_index)){ /* ... input ... */
            sint32 input_past_reach = partial.inside_indices(input_synapse_index + input_synapse_index_offset).reach_past_loops();
            sint32 input_index = partial.inside_indices(input_synapse_index + input_synapse_index_offset).starts();
            if(SynapseIterator<>::is_index_input(input_index)){
              input_index = SynapseIterator<>::input_index_from_synapse_index(input_index - input_start_offset);
              input_past_reach = partial_input_synapses.reach_past_loops<InputSynapseInterval>(input_index);
              input_index = partial_input_synapses[input_index];
              if(SynapseIterator<>::is_index_input(input_index)){
                input_index = SynapseIterator<>::input_index_from_synapse_index(input_index);
                assert( 0 == input_past_reach );
                inner_neuron_input_weight_pairs += std::string("(")
                /* input */+ " inputs[input_start + " + std::to_string(input_index) + "]"
                /* weight */+ " * inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
                + ") + ";
              }else{
                if(0 != input_past_reach){
                  inner_neuron_input_weight_pairs += past_reach_guard(input_past_reach, std::string("(")
                  /* input */+ " outputs[output_start + " + std::to_string(input_index - static_cast<sint32>(input_past_reach * solution.neuron_number())) + "]"
                  /* weight */+ " * inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
                  + ")") + std::string(" + ");
                }else{ /* input doesn't reach to the past */
                  inner_neuron_input_weight_pairs += std::string("(")
                  /* input */+ " outputs[output_start + " + std::to_string(input_index) + "]"
                  /* weight */+ " * inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
                  + ") + ";
                }
              }
            }else{
              input_index = input_index + input_start_offset;
              if(0 != input_past_reach){
                inner_neuron_input_weight_pairs += past_reach_guard(input_past_reach, std::string("(")
                /* input */+ " outputs[output_start + " + std::to_string(input_index - static_cast<sint32>(input_past_reach * solution.neuron_number())) + "]"
                /* weight */+ " * inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
                + ")") + std::string(" + ");
              }else{ /* input doesn't reach to the past */
                inner_neuron_input_weight_pairs += std::string("(")
                /* input */+ " outputs[output_start + " + std::to_string(input_index) + "]"
                /* weight */+ " * inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
                + ") + ";
              }
            }
            ++input_start_offset;
            if(input_start_offset >= partial.inside_indices(input_synapse_index + input_synapse_index_offset).interval_size()){
              input_start_offset = 0;
              ++input_synapse_index_offset;
            }
          }else{ /* ... bias ... */
            inner_neuron_input_weight_pairs += std::string("(")
            /* bias weight */+ "inputs[" + std::to_string(weight_table_offset + weight_index) + "]"
            + ") + ";
          }
        }
      }, weight_synapse_start, partial.weight_synapse_number(inner_neuron_index));
      inner_neuron_input_weight_pairs += "0.0 \n);\n";
      inner_neuron_operation += inner_neuron_input_weight_pairs;
      inner_neuron_operation += "neuron_partial_result = " + transfer_function.get_cl_function_for(
        partial.neuron_transfer_functions(inner_neuron_index),
        "neuron_partial_result"
      )+";\n";
      inner_neuron_operation += (
        "outputs[output_start + " + std::to_string(partial.output_data().starts() + inner_neuron_index) + "] = (\n"
        + SpikeFunction::get_cl_function_for(
          "inputs[" + std::to_string(weight_table_offset + spike_weight_index)+ "]"/*parameter*/,
          "neuron_partial_result"/* new_data */,
          past_reach_guard(
            1u, std::string("(")
            + "outputs[output_start + " + std::to_string(partial.output_data().starts() + inner_neuron_index) + " - neuron_array_size]"
            + ")"
          )/* previous_data */
        )+"\n);\n"
      );
      weight_synapse_start += partial.weight_synapse_number(inner_neuron_index);
      input_synapse_index += input_synapse_index_offset;
      input_synapse_index_offset = 0u;
      inner_neuron_operation += "\n";
      neuron_operations += inner_neuron_operation;
    }/*for(each inner neuron)*/

    if(0 < partial.solved_features_size()){ /* if the partial solves any feature */
      for(const FeatureGroup& feature : partial.solved_features()){
        RafkoNetFeatureExecutor::add_kernel_code_to(neuron_operations, feature, "output_start", !feature_locals_declared);
        feature_locals_declared = true;
      }
    }

    weight_table_offset += partial.weight_table_size();
    /*!Note: Because each partial has a different weight table, an offset is required to keep track in case there are multiple partial solutions
     * inside the solution.
     */
  }/*for(every partial in the solution)*/
  source_base = std::regex_replace(source_base, std::regex("\\$\\$name\\$\\$"), name);
  source_base = std::regex_replace(source_base, std::regex("\\$\\$neuron_array_size\\$\\$"), std::to_string(solution.neuron_number()));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$weight_table_offset\\$\\$"), std::to_string(weight_table_offset));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$neuron_memory_size\\$\\$"), std::to_string(solution.network_memory_length()));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$sequence_size\\$\\$"), std::to_string(sequence_size));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$neuron_operations\\$\\$"), neuron_operations);
  source_base = std::regex_replace(source_base, std::regex("\\$\\$prefill_input_num\\$\\$"), std::to_string(prefill_input_num));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$network_input_size\\$\\$"), std::to_string(solution.network_input_size()));
  return source_base;
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_net */
