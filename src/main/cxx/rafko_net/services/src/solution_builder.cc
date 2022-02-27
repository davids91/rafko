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
#include <map>
#include <regex>
#include <functional>

#include "rafko_utilities/models/rafko_gpu_kernel_library.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/models/transfer_function.h"
#include "rafko_net/models/spike_function.h"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_net/models/input_function.h"
#include "rafko_net/models/neuron_info.h"
#include "rafko_net/services/neuron_router.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_network_feature.h"

#include "rafko_net/services/partial_solution_builder.h"

namespace rafko_net{

std::unique_ptr<Solution> SolutionBuilder::build(const RafkoNet& net, bool optimize_to_gpu){
  NeuronRouter neuron_router(net);
  std::unique_ptr<Solution> solution = std::make_unique<Solution>();
  std::uint32_t overall_partial_solution_count = 0u;
  double remaining_megabytes_in_row = 0;
  double current_neuron_megabyte_size;
  std::uint32_t reach_back_max = 0u;
  std::uint32_t reach_index_max = 0u;
  std::uint32_t current_neuron_index;
  bool has_neuron = false;

  if(0 == net.output_neuron_number()) throw std::runtime_error("Can't build a solution with 0 output Neurons!");
  while(!neuron_router.finished()){ /* Until the whole network is processed */
    if( (!optimize_to_gpu)&&(0 == solution->cols_size()) )
      neuron_router.collect_subset(settings.get_max_solve_threads(),settings.get_device_max_megabytes(), false);
    else neuron_router.collect_subset(settings.get_max_solve_threads(),settings.get_device_max_megabytes(), true);

    remaining_megabytes_in_row = settings.get_device_max_megabytes();
    const double max_megabytes_in_one_partial = ( remaining_megabytes_in_row / static_cast<double>(settings.get_max_solve_threads()) );
    overall_partial_solution_count = solution->partial_solutions_size();

    if(0u < neuron_router.get_subset_size()){
      for(std::uint32_t partial_index_in_row = 0; partial_index_in_row < settings.get_max_solve_threads(); ++partial_index_in_row){
        if(nullptr == settings.get_arena_ptr() ) *solution->add_partial_solutions() = PartialSolution();
        else *solution->add_partial_solutions() = *google::protobuf::Arena::CreateMessage<PartialSolution>(settings.get_arena_ptr());

        /* fill up the partial with Neurons */
        PartialSolution& this_partial = *solution->mutable_partial_solutions(solution->partial_solutions_size()-1);
        double remaining_megabytes_in_partial = max_megabytes_in_one_partial;
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
          if(!has_neuron) /* A new Neuron index was acquired from the router, refresh size info */
            current_neuron_megabyte_size = NeuronInfo::get_neuron_estimated_size_megabytes(net.neuron_array(current_neuron_index));

          if(current_neuron_megabyte_size >= remaining_megabytes_in_partial)
            break;

          if(0u == this_partial.output_data().interval_size()) /* The first Neuron inside the partial solution shall determine its start */
            this_partial.mutable_output_data()->set_starts(current_neuron_index);

          std::pair<std::uint32_t,std::uint32_t> neuron_input_params = PartialSolutionBuilder::add_neuron_to_partial_solution(
            net, current_neuron_index, this_partial
          );
          remaining_megabytes_in_row -= current_neuron_megabyte_size;
          remaining_megabytes_in_partial -= current_neuron_megabyte_size;

          if(reach_back_max < std::get<0>(neuron_input_params))
            reach_back_max = std::get<0>(neuron_input_params);

          if(reach_index_max < std::get<1>(neuron_input_params))
            reach_index_max = std::get<1>(neuron_input_params);

          std::vector<std::reference_wrapper<const FeatureGroup>> features_solved_by_neuron = neuron_router.confirm_first_subset_element_processed(current_neuron_index);
          for(const FeatureGroup& feature : features_solved_by_neuron){
            *this_partial.add_solved_features() = feature;
          }

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

      if(solution->partial_solutions_size() > static_cast<std::int32_t>(overall_partial_solution_count))
        solution->add_cols(solution->partial_solutions_size() - overall_partial_solution_count);
      neuron_router.reset_remaining_subset(); /* Whichever Neuron coudn't fit into the partial shall have its state reset */
    } /* if(0u < neuron_router.get_subset_size()) */
  } /* while(!neuron_router.finished()) */

  for(PartialSolution& partial : *solution->mutable_partial_solutions()){
    std::sort(partial.mutable_solved_features()->begin(),partial.mutable_solved_features()->end(),
    [](const FeatureGroup& a, const FeatureGroup& b){
      return a.feature() < b.feature();
    }); /*!Note: Sorting out FeatureGroups to enforce dependencies, where the larger enum values must be executed later */
  }

  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_back_max + 1u); /* Current loop is "0" reachback, so length should be at least 1 */
  solution->set_network_input_size(reach_index_max + 1u);
  assert( net.input_data_size() == reach_index_max + 1u);

  return solution;
}

#if(RAFKO_USES_OPENCL)
std::string SolutionBuilder::get_kernel_for_solution(
  const Solution& solution, std::string name, std::uint32_t sequence_size, std::uint32_t prefill_input_num,
  const rafko_mainframe::RafkoSettings& settings
){
  std::string input_functions_kernel = R"(
    switch(operation->behavior_index){
      case neuron_input_function_add
    }
  )";
  std::string transfer_functions_kernel;
  std::string spike_functions_kernel;

  /*!Note: Network solve starts from the first memory/sequence slot of the outputs buffer
   * which is calculated from get_global_id(0) and the size of the max(sequence, neuron_memory);
   * UNLESS mode variable input is non-zero.
   * If the mode varaible is non-zero the start of the execution shall start at the end of the
   * last output slice ( the last neuron value array the kernel is supposed to be updating )
   * and every other output slice before that is considered outputs from the past
   */
  std::string source_base = rafko_utilities::random_function + rafko_utilities::atomic_double_add_function
   + R"(
    typedef enum rafko_neuron_action_e{
      neuron_action_set_by_input = 0,
      neuron_action_set_by_neuron,
      neuron_action_set_by_past, /* ...by_past implicitly suggests that it's coming from a past run of a neuron*/
      neuron_action_input_function_by_input,
      neuron_action_input_function_by_neuron,
      neuron_action_input_function_by_past,
      neuron_action_input_function_bias,
      neuron_action_transfer_function,
      neuron_action_spike_function
    }rafko_neuron_action_t __attribute__ ((aligned));

    ==input_function_enums==
    ==transfer_function_enums==
    ==spike_function_enums==

    typedef struct rafko_neuron_operation_s{
      int global_buffer_index[2]; /* Index values pointing to elements in the global input/output array */
      rafko_neuron_action_t action;
      unsigned char behavior_index;
    } rafko_neuron_operation_t __attribute__ ((aligned));

    void neuron_operation(
      rafko_neuron_action_t action, unsigned char behavior_index,
      __constant double* weight, __constant double* input_data, __global double* neuron_data, __private double* local_buffer
    ){
      const double weight_value = (NULL!=weight)?(*weight):(0.0); /* in case the weight is not set, it is to be taken as 0.0 */
      /*!Note: weight should only be NULL in case of a transfer function, where no weight is used */
      double input_value;
      if(NULL != input_data)input_value = *input_data;
        else if(NULL != neuron_data)input_value = *neuron_data;
        else if(action == neuron_action_input_function_bias) input_value = 1.0;
        else input_value = 0.0;
      /*!Note: in case the input data is not set, and the neuron data is also not set input value shall be 0.0,
       * it happens, when:
       * - the bias operation is requested: the weight value needs to be multiplied by 1.0 to keep its value
       * - neuron input function is called on an input which is in an unreachable past; value should be 0.0
       * - neuron spike function is called during the first run of the network; the value to be used should be 0.0
       */

      switch(action){
        case neuron_action_set_by_input:
        case neuron_action_set_by_neuron:
        case neuron_action_set_by_past:
          *local_buffer = (weight_value * input_value);
          break;
        case neuron_action_input_function_by_input:
        case neuron_action_input_function_by_neuron:
        case neuron_action_input_function_bias:
        case neuron_action_input_function_by_past:
          ==input_functions_kernel==
          break;
        case neuron_action_transfer_function:
          ==transfer_functions_kernel==
          break;
        case neuron_action_spike_function:
          ==spike_functions_kernel==
          *local_buffer = 0.0; /* zero out local_buffer for the next Neuron */
          break;
        default: break;
      }
    }

    __constant const rafko_neuron_operation_t arr[] = {
      /* +++ GENERATED_NEURON_CODE +++ */
      ==neuron_operations==
      /* --- GENERATED_NEURON_CODE --- */
    };

    void kernel ==name==(
       __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
       __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      double local_buffer = 0.0;
      unsigned char behavior_index;
      __constant double* input_weight;
      __constant double* input_from_inputs;
      __global double* input_from_neurons;
      if( (input_sizes_size == 3) && (output_sizes_size == 2) ){
        const int sequence_index = get_global_id(0);
        const int neuron_array_size = ==neuron_array_size==;
        const int neuron_memory_slots = max(2, ==neuron_memory_size==);
        const int max_backreach = (neuron_memory_slots - 1);
        const int sequence_size = ==sequence_size==;
        const int prefill_input_num = ==prefill_input_num==;
        const int network_input_size = ==network_input_size==;
        int output_start;
        int current_max_backreach;
        int input_start;
        int label_index = 0; /* index inside the current inside sequence */
        int sequence_start;
        int sequence_max_index;
        bool evaluate_network = (inputs[0] == 0);
        if(evaluate_network){ /* normal evaluation */
          current_max_backreach = 0;
          input_start = ==weight_table_offset== + (sequence_index * (sequence_size + prefill_input_num) * network_input_size);
          output_start = sequence_index * (sequence_size + prefill_input_num) * neuron_array_size;
          sequence_start = 0;
          sequence_max_index = max(( sequence_size + prefill_input_num ), 1) - 1;
        }else{ /* run-once with memory */
          current_max_backreach = max_backreach;
          input_start = ==weight_table_offset==;
          output_start = (max_backreach * neuron_array_size);
          sequence_start = max(max(max_backreach,( sequence_size + prefill_input_num )), 1) - 1;
          sequence_max_index = sequence_start;
        }
        for(int label_index = sequence_start; label_index <= sequence_max_index; ++label_index){
          double neuron_partial_result;
          int neuron_operation_index = 0;
          while(neuron_operation_index < ==neuron_operation_number==){
            if(arr[neuron_operation_index].action == neuron_action_transfer_function){
              input_weight = NULL;
            }else{
              input_weight = &inputs[arr[neuron_operation_index].global_buffer_index[0]];
            }
            input_from_neurons = NULL;
            input_from_inputs = NULL;
            switch(arr[neuron_operation_index].action){
              case neuron_action_set_by_input:
              case neuron_action_input_function_by_input:
                input_from_inputs = &inputs[input_start + arr[neuron_operation_index].global_buffer_index[1]];
                break;
              case neuron_action_set_by_neuron:
              case neuron_action_input_function_by_neuron:
                input_from_neurons = &outputs[output_start + arr[neuron_operation_index].global_buffer_index[1]];
                break;
              case neuron_action_set_by_past:
              case neuron_action_input_function_by_past:{
                if( ((arr[neuron_operation_index].behavior_index & 0xF0u) >> 8u)/*past_index*/ < current_max_backreach )
                  input_from_neurons = &outputs[output_start + arr[neuron_operation_index].global_buffer_index[1]]; /* Neuron index correction to take the past value is hardwired into the command array */
              }break;
              case neuron_action_spike_function:
                /* When the first run is in progress, the spike function should not know about the previous value, otherwise the value is set by the command array */
                if( ((sequence_start - label_index) > 0)||(!evaluate_network) )
                  input_from_neurons = &outputs[output_start + arr[neuron_operation_index].global_buffer_index[1]];
                break;
              case neuron_action_transfer_function: /* transfer function uses local buffer only */
              case neuron_action_input_function_bias: /* bias doesn't use the input_from_neurons or input_from_inputs, only weight */
              default: break;
            }
            neuron_operation(
              arr[neuron_operation_index].action, (arr[neuron_operation_index].behavior_index & 0xFu),
              input_weight, input_from_inputs, input_from_neurons, &local_buffer
            );
            //=/=neuron_group_features== //TODO: check features not in every loop, maybe just after every row
            ++neuron_operation_index;
          }/*for(neuron operations)*/
          input_start += network_input_size;
          output_start += neuron_array_size;
          if(current_max_backreach < max_backreach)
            ++current_max_backreach;
        }/*for(every relevant label)*/

        outputs[output_sizes[0]] = 0.0; /* zero out performance error */
        ==performance_operations==

      }/*if(input sizes match)*/
    }/* kernel */
  )";
  TransferFunction transfer_function(settings);
  std::uint32_t weight_table_offset = 1u; /* the first input number is the mode of the solution, so weight table starts after that */
  /*!Note: Weight table is supposed to consist of the weight tables of the partial solutions
   * in the order they are inside the @Solution. Inside the kernel this order is used
   * to query the wieghts by position in the device weight table.
   * This note is strictly for one kernel solutions! Networks spanning multiple devices
   * are to re-think this logic.
   */
  std::string neuron_operations;
  std::function<std::string(std::uint32_t, std::string)> past_reach_guard = [](std::uint32_t past_reach, std::string content){
    return "( (min(current_max_backreach,max_backreach) < " + std::to_string(past_reach) + " )?(0.0):( " + content + ") )";
  };
  std::function<std::string(std::string)> label_reach_guard = [](std::string content){
    return "(  ( ((sequence_start - label_index) == 0)&&(evaluate_network) )?(0.0):( " + content + ")  )";
  };
  /*!Note: When the network has no memory, backreach counters can not be used to reach the previous value of the Neuron
   * so it would be best to use the currently evaluated label for it (every label except the first reaches the Neurons previous value)
   * EXCEPT it renders single runs incorrect, so it can only be used if the network is under evaluation; since single runs already have
   * the adequate memory allocated, and require no restrictions
   */
  bool feature_locals_declared = false;
  std::vector<std::reference_wrapper<const FeatureGroup>> performance_feature_list;
  std::uint32_t neuron_operation_number = 0u;
  for(const PartialSolution& partial : solution.partial_solutions()){
    SynapseIterator<InputSynapseInterval> partial_input_synapses(partial.input_data());
    std::uint32_t input_synapse_index = 0u;
    std::uint32_t input_synapse_index_offset = 0u;
    std::uint32_t input_start_offset = 0u;
    std::uint32_t weight_synapse_start = 0u;

    for(std::uint32_t inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
      bool first_weight_in_neuron = true;
      bool first_input_in_neuron = true;
      std::uint32_t spike_weight_index;
      SynapseIterator<>::iterate(partial.weight_indices(),[&](std::int32_t weight_index){
        if(first_weight_in_neuron){
          first_weight_in_neuron = false;
          spike_weight_index = weight_index;
        }else{
          if(input_synapse_index_offset < partial.index_synapse_number(inner_neuron_index)){ /* ... input ... */
            std::int32_t input_past_reach = partial.inside_indices(input_synapse_index + input_synapse_index_offset).reach_past_loops();
            std::int32_t input_index = partial.inside_indices(input_synapse_index + input_synapse_index_offset).starts();
            if(SynapseIterator<>::is_index_input(input_index)){ /* Neuron input points to partial input */
              input_index = SynapseIterator<>::input_index_from_synapse_index(input_index - input_start_offset);
              input_past_reach = partial_input_synapses.reach_past_loops<InputSynapseInterval>(input_index);
              input_index = partial_input_synapses[input_index];
              if(SynapseIterator<>::is_index_input(input_index)){ /* Partial input points to Network input */
                input_index = SynapseIterator<>::input_index_from_synapse_index(input_index);
                assert( 0 == input_past_reach );
                std::string input_operation = "neuron_action_input_function_by_input";
                if(first_input_in_neuron) input_operation = "neuron_action_set_by_input";
                neuron_operations += std::string("\t{ ")
                + "{"
                  + std::to_string(weight_table_offset + weight_index) + ","
                  + std::to_string(input_index)
                + "} /*global_buffer:weight_index,input_index*/, "
                + input_operation + std::string("/*action*/, ")
                + "0 /*past_index(which is irrelevant in this operation)*/ "
                + std::string("}, \n");
              }else{ /* Partial input points to a Neuron */
                if(0 != input_past_reach){
                  std::string input_operation = "neuron_action_input_function_by_past";
                  if(first_input_in_neuron) input_operation = "neuron_action_set_by_past";
                  std::string operation_action = InputFunction::get_kernel_enum_for(partial.neuron_input_functions(inner_neuron_index));
                  neuron_operations += std::string("\t{ ")
                  + "{"
                    + std::to_string(weight_table_offset + weight_index) + ","
                    + "(" + std::to_string(input_index - static_cast<std::int32_t>(input_past_reach * solution.neuron_number())) + ")"
                  + "} /*global_buffer:weight_index,neuron_index*/, "
                  + input_operation + std::string("/*action*/, ")
                  + "( ((" +std::to_string(input_past_reach)+ "&0xFu) << 8u)|(" + operation_action + "))/*past_index + input function index*/"
                  + std::string("}, \n");
                }else{ /* input doesn't reach to the past */
                  std::string input_operation = "neuron_action_input_function_by_neuron";
                  if(first_input_in_neuron) input_operation = "neuron_action_set_by_neuron";
                  std::string operation_action = InputFunction::get_kernel_enum_for(partial.neuron_input_functions(inner_neuron_index));
                  neuron_operations += std::string("\t{ ")
                  + "{"
                    + std::to_string(weight_table_offset + weight_index) + ","
                    + "(" + std::to_string(input_index) + ")"
                  + "} /*global_buffer:weight_index,neuron_index*/, "
                  + input_operation + std::string("/*action*/, ")
                  + operation_action + "/*input_function*/ "
                  + std::string("}, \n");
                }
              }
            }else{ /* Neuron input index does not point to the partial input; it's definitely points inside the network */
              input_index = input_index + input_start_offset;
              if(0 != input_past_reach){
                std::string input_operation = "neuron_action_input_function_by_past";
                if(first_input_in_neuron) input_operation = "neuron_action_set_by_past";
                std::string operation_action = InputFunction::get_kernel_enum_for(partial.neuron_input_functions(inner_neuron_index));
                neuron_operations += std::string("\t{ ")
                + "{"
                  + std::to_string(weight_table_offset + weight_index) + ","
                  + "(" + std::to_string(input_index - static_cast<std::int32_t>(input_past_reach * solution.neuron_number())) + ")"
                + "} /*global_buffer:weight_index,neuron_index*/, "
                + input_operation + std::string("/*action*/, ")
                + "( ((" +std::to_string(input_past_reach)+ "&0xFu) << 8u)|(" + operation_action + "))/*past_index + input function index*/"
                + std::string("}, \n");
              }else{ /* input doesn't reach to the past */
                std::string input_operation = "neuron_action_input_function_by_neuron";
                if(first_input_in_neuron) input_operation = "neuron_action_set_by_neuron";
                std::string operation_action = InputFunction::get_kernel_enum_for(partial.neuron_input_functions(inner_neuron_index));
                neuron_operations += std::string("\t{ ")
                + "{"
                  + std::to_string(weight_table_offset + weight_index) + ","
                  + "(" + std::to_string(input_index) + ")"
                + "} /*global_buffer:weight_index,neuron_index*/, "
                + input_operation + std::string("/*action*/, ")
                + operation_action + "/*input_function*/ "
                + std::string("}, \n");
              }
            }
            ++input_start_offset;
            if(input_start_offset >= partial.inside_indices(input_synapse_index + input_synapse_index_offset).interval_size()){
              input_start_offset = 0;
              ++input_synapse_index_offset;
            }
          }else{ /* ... bias ... */
            std::string operation_action = InputFunction::get_kernel_enum_for(partial.neuron_input_functions(inner_neuron_index));
            neuron_operations += std::string("\t{ ")
            + "{"
              + std::to_string(weight_table_offset + weight_index) + ",0/* second input doesn't matter with bias*/"
            + "} /*global_buffer:bias_index,-*/, "
            + std::string("neuron_action_input_function_bias /*action*/, ")
            + operation_action + "/*input_function*/ "
            + std::string("}, \n");
          }
          first_input_in_neuron = false;
        }
        ++neuron_operation_number;
      }, weight_synapse_start, partial.weight_synapse_number(inner_neuron_index));
      neuron_operations += std::string("{ ")
      + "\t{"
        + "0,0 /* none of these matter with the transfer function, as it uses local buffer only */"
      + "} /*global_buffer:-,-*/, "
      + std::string("neuron_action_transfer_function /*action*/, ")
      + transfer_function.get_kernel_enum_for(partial.neuron_transfer_functions(inner_neuron_index)) + "/*transfer_function index*/ "
      + std::string("}, \n");
      ++neuron_operation_number;

      neuron_operations += std::string("{ ")
      + "\t{"
        + std::to_string(weight_table_offset + spike_weight_index) + ","
        + "(" + std::to_string(partial.output_data().starts() + inner_neuron_index) + ")"
      + "} /*global_buffer:weight_index,target neuron_index*/, "
      + std::string("neuron_action_spike_function/*action*/, ")
      + SpikeFunction::get_kernel_enum_for(spike_function_memory) + "/*spike_function index*/ "
      + std::string("}, \n");
      ++neuron_operation_number;
      weight_synapse_start += partial.weight_synapse_number(inner_neuron_index);
      input_synapse_index += input_synapse_index_offset;
      input_synapse_index_offset = 0u;
    }/*for(each inner neuron)*/

    //TODO: Re-add features if(0 < partial.solved_features_size()){ /* if the partial solves any feature */
    //   for(const FeatureGroup& feature : partial.solved_features()){
    //     if(NeuronInfo::is_feature_relevant_to_solution(feature.feature())){
    //       RafkoNetworkFeature::add_kernel_code_to(
    //         neuron_operations, feature, settings, solution, "", "output_start", !feature_locals_declared
    //       );
    //       feature_locals_declared = true;
    //     }else if(NeuronInfo::is_feature_relevant_to_performance(feature.feature())){
    //       performance_feature_list.push_back(feature);
    //     }
    //   }
    // }

    weight_table_offset += partial.weight_table_size();
    /*!Note: Because each partial has a different weight table, an offset is required to keep track in case there are multiple partial solutions
     * inside the solution.
     */
  }/*for(every partial in the solution)*/
  std::string performance_operations;
  std::set<Neuron_group_features> already_declared_locals;
  for(const FeatureGroup& feature : performance_feature_list){
    bool declare_locals = ( /* if the locals have not been declared yet */
      already_declared_locals.end() == already_declared_locals.find(feature.feature())
    );
    RafkoNetworkFeature::add_kernel_code_to(
      performance_operations, feature, settings, solution,
      "1u"/*input_start_index:weight table start*/, "output_sizes[0]"/*output_start_index: last output*/,
      declare_locals
    );

    if(declare_locals)
     already_declared_locals.insert(feature.feature());
  }
  source_base = std::regex_replace(source_base, std::regex("==name=="), name);
  source_base = std::regex_replace(source_base, std::regex("==neuron_array_size=="), std::to_string(solution.neuron_number()));
  source_base = std::regex_replace(source_base, std::regex("==weight_table_offset=="), std::to_string(weight_table_offset));
  source_base = std::regex_replace(source_base, std::regex("==neuron_memory_size=="), std::to_string(solution.network_memory_length()));
  source_base = std::regex_replace(source_base, std::regex("==sequence_size=="), std::to_string(sequence_size));
  source_base = std::regex_replace(source_base, std::regex("==neuron_operations=="), neuron_operations);
  source_base = std::regex_replace(source_base, std::regex("==prefill_input_num=="), std::to_string(prefill_input_num));
  source_base = std::regex_replace(source_base, std::regex("==network_input_size=="), std::to_string(solution.network_input_size()));
  source_base = std::regex_replace(source_base, std::regex("==performance_operations=="), performance_operations);

  source_base = std::regex_replace(source_base, std::regex("==neuron_operation_number=="), std::to_string(neuron_operation_number));
  source_base = std::regex_replace(source_base, std::regex("==input_function_enums=="), InputFunction::get_kernel_enums());
  source_base = std::regex_replace(source_base, std::regex("==transfer_function_enums=="), TransferFunction::get_kernel_enums());
  source_base = std::regex_replace(source_base, std::regex("==spike_function_enums=="), SpikeFunction::get_kernel_enums());

  source_base = std::regex_replace(
    source_base, std::regex("==input_functions_kernel=="),
    InputFunction::get_kernel_function_for("behavior_index","*local_buffer","(weight_value * input_value)")
  );
  source_base = std::regex_replace(
    source_base, std::regex("==transfer_functions_kernel=="),
    transfer_function.get_kernel_function_for("behavior_index","*local_buffer","*local_buffer")
  );
  source_base = std::regex_replace(
    source_base, std::regex("==spike_functions_kernel=="),
    SpikeFunction::get_kernel_function_for("behavior_index","weight_value","*neuron_data","*local_buffer")
  );
  std::cout << "Agent code: " << source_base << std::endl;
  return source_base;
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_net */
