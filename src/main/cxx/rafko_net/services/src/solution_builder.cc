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

#include "rafko_net/services/solution_builder.hpp"

#include <math.h>
#include <memory>
#include <stdexcept>
#include <utility>

#if(RAFKO_USES_OPENCL)
#include <map>
#include <regex>
#include <functional>

#include "rafko_utilities/models/rafko_gpu_kernel_library.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_net/models/transfer_function.hpp"
#include "rafko_net/models/spike_function.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_net/models/input_function.hpp"
#include "rafko_net/models/neuron_info.hpp"
#include "rafko_net/services/neuron_router.hpp"
#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_net/services/rafko_network_feature.hpp"

#include "rafko_net/services/partial_solution_builder.hpp"

namespace rafko_net{

void SolutionBuilder::update(Solution* previous, const RafkoNet& network, bool optimize_to_gpu){
  RFASSERT(nullptr != previous);
  Solution* solution = build(network, nullptr, optimize_to_gpu);
  solution->Swap(previous);
  delete solution;
}

Solution* SolutionBuilder::build(const RafkoNet& net, google::protobuf::Arena* arena_ptr, bool optimize_to_gpu){
  RFASSERT_SCOPE(SOLUTION_BUILD);
  NeuronRouter neuron_router(net);
  Solution* solution = google::protobuf::Arena::CreateMessage<Solution>(arena_ptr);
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
      neuron_router.collect_subset(m_settings.get_max_solve_threads(), m_settings.get_device_max_megabytes(), false);
    else neuron_router.collect_subset(m_settings.get_max_solve_threads(),m_settings.get_device_max_megabytes(), true);

    remaining_megabytes_in_row = m_settings.get_device_max_megabytes();
    const double max_megabytes_in_one_partial = ( remaining_megabytes_in_row / static_cast<double>(m_settings.get_max_solve_threads()) );
    overall_partial_solution_count = solution->partial_solutions_size();

    if(0u < neuron_router.get_subset_size()){
      for(std::uint32_t partial_index_in_row = 0; partial_index_in_row < m_settings.get_max_solve_threads(); ++partial_index_in_row){
        /* Add a partial solution and fill it up with Neurons */
        PartialSolution& this_partial = *solution->add_partial_solutions();
        PartialSolutionBuilder partial_builder(this_partial);
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

          std::pair<std::uint32_t,std::uint32_t> neuron_input_params = partial_builder.add_neuron_to_partial_solution(net, current_neuron_index);
          remaining_megabytes_in_row -= current_neuron_megabyte_size;
          remaining_megabytes_in_partial -= current_neuron_megabyte_size;

          if(reach_back_max < std::get<0>(neuron_input_params))
            reach_back_max = std::get<0>(neuron_input_params);

          if(reach_index_max < std::get<1>(neuron_input_params))
            reach_index_max = std::get<1>(neuron_input_params);

          std::vector<std::uint32_t> features_solved_by_neuron = neuron_router.confirm_first_subset_element_processed(current_neuron_index);
          for(std::uint32_t feature_index : features_solved_by_neuron){
            *this_partial.add_solved_features() = net.neuron_group_features(feature_index);
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

  RFASSERT_LOG("Solution has {} partials!", solution->partial_solutions_size());
  solution->set_output_neuron_number(net.output_neuron_number());
  solution->set_neuron_number(net.neuron_array_size());
  solution->set_network_memory_length(reach_back_max + 1u); /* Current loop is "0" reachback, so length should be at least 1 */
  solution->set_network_input_size(reach_index_max + 1u);
  RFASSERT( net.input_data_size() == reach_index_max + 1u);
  RFASSERT( net.memory_size() == solution->network_memory_length());
  return solution;
}

#if(RAFKO_USES_OPENCL)
std::string SolutionBuilder::get_kernel_for_solution(
  const Solution& solution, std::string name, std::uint32_t sequence_size, std::uint32_t prefill_input_num,
  const rafko_mainframe::RafkoSettings& settings
){
  RFASSERT_LOG("Building GPU kernel for solution");
  /*!Note: Network solve starts from the first memory/sequence slot of the outputs buffer
   * which is calculated from get_global_id(0) and the size of the max(sequence, neuron_memory);
   * UNLESS mode variable input is non-zero.
   * If the mode varaible is non-zero the start of the execution shall start at the end of the
   * last output slice ( the last neuron value array the kernel is supposed to be updating )
   * and every other output slice before that is considered outputs from the past.
   * The value of the MODE value also tells how many times to run the network.
   */
  std::string source_base = rafko_utilities::random_function + rafko_utilities::atomic_double_add_function
   + R"(
    void __kernel ==name==(
       __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
       __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      if( (input_sizes_size == 3) && (output_sizes_size == 2) ){
        const int sequence_index = get_global_id(0);
        const int neuron_array_size = ==neuron_array_size==;
        const int neuron_memory_slots = max(2, ==network_memory_slots==); /* 2 slots are a minimum because of the Spike function */
        const int sequence_size = ==sequence_size==;
        const int prefill_input_num = ==prefill_input_num==;
        const int network_input_size = ==network_input_size==;
        const int inputs_in_one_sequence = sequence_size + prefill_input_num;
        const int outputs_in_one_sequence = max(neuron_memory_slots, inputs_in_one_sequence);
        const int eval_start = neuron_memory_slots - min(neuron_memory_slots, inputs_in_one_sequence);
        const int received_run_count = inputs[0];

        int input_start = ==weight_table_offset== + (sequence_index * inputs_in_one_sequence * network_input_size);
        int output_start = sequence_index * outputs_in_one_sequence * neuron_array_size;
        int current_max_backreach;
        int current_slot_index; /* index inside the current sequence */
        int run_count;
        bool evaluate_network = (received_run_count == 0);
        if(evaluate_network){ /* normal evaluation */
          current_max_backreach = 0;
          current_slot_index = eval_start;
          output_start += current_slot_index * neuron_array_size; /* correct output start in case network memory is more, than the environment sequence size */
          run_count = inputs_in_one_sequence;
        }else{ /* run with memory */
          current_max_backreach = neuron_memory_slots;
          current_slot_index = max(outputs_in_one_sequence, 1) - 1;
          run_count = min(received_run_count, outputs_in_one_sequence);
          output_start += current_slot_index * neuron_array_size; /* correct output start by putting it to the last label in sequence */
        }
        const int starting_slot_index = current_slot_index;

        #pragma unroll
        for(int run_index = 0; run_index < run_count; ++run_index){
          double neuron_partial_result;
          /* +++ GENERATED_NEURON_CODE +++ */
          ==neuron_operations==
          /* --- GENERATED_NEURON_CODE --- */

          if(
            !evaluate_network /* If network is not evaluating, the current run needs to be copied one slot down */
            &&((run_index < (run_count-1)) || (1 == run_count))
          ){
            int sequence_start = sequence_index * outputs_in_one_sequence * neuron_array_size;
            for(int slot_index = 0; slot_index < (outputs_in_one_sequence - 1); ++slot_index){
              for(int data_index = 0; data_index < neuron_array_size; ++data_index){
                outputs[sequence_start + data_index] = outputs[sequence_start + neuron_array_size + data_index];
              }
              sequence_start += neuron_array_size;
            }
          }
          input_start += network_input_size;
          if(evaluate_network){ /* No need to update these when the network is not under evaluation, as then the same slot is being updated always */
            output_start += neuron_array_size;
            ++current_slot_index;
            if(current_max_backreach < neuron_memory_slots)
              ++current_max_backreach;
          }
        }

        if(0 == get_global_id(0))outputs[output_sizes[0]] = 0.0; /* zero out performance error */
        if(evaluate_network){
          while(!isequal(0.0, outputs[output_sizes[0]])); /* don't judge me, there's no global barrier */
          ==performance_operations==
        }

      }/*if(input sizes match)*/
    }/* kernel */
  )";
  source_base = std::regex_replace(source_base, std::regex("==output_start=="), std::to_string(
    (solution.network_memory_length()-1) * solution.neuron_number()
  ));
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
    return "( (current_max_backreach < " + std::to_string(past_reach) + " )?(0.0):( " + content + ") )";
  };

  /*!Note: When the network has no memory, backreach counters can not be used to reach the previous value of the Neuron
   * so it would be best to use the currently evaluated label for it (every label except the first reaches the Neurons previous value)
   * EXCEPT it renders single runs incorrect, so it can only be used if the network is under evaluation; since single runs already have
   * the adequate memory allocated, and require no restrictions
   */
  bool feature_locals_declared = false;
  std::vector<std::reference_wrapper<const FeatureGroup>> performance_feature_list;
  for(const PartialSolution& partial : solution.partial_solutions()){
    SynapseIterator<InputSynapseInterval> partial_input_synapses(partial.input_data());
    std::uint32_t input_synapse_index = 0u;
    std::uint32_t input_synapse_index_offset = 0u;
    std::uint32_t input_offset_in_current_synapse = 0u;
    std::uint32_t weight_synapse_start = 0u;

    for(std::uint32_t inner_neuron_index = 0; inner_neuron_index < partial.output_data().interval_size(); ++inner_neuron_index){
      bool first_weight_synapse_in_neuron = true;
      bool first_input_in_neuron = true;
      std::function<std::string(std::string,std::string,bool)> input_function_lambda =
      [&inner_neuron_index, &partial](std::string sum, std::string b, bool first)->std::string{
        if(first)return sum + " = " + b + ";"; /* The first operation shall set the value, not update it */
        else return(
          sum + "=" + InputFunction::get_kernel_function_for(partial.neuron_input_functions(inner_neuron_index), sum, b) + ";\n"
        );
      };
      std::uint32_t spike_weight_index;
      std::string inner_neuron_operation = "";
      SynapseIterator<>::skim(partial.weight_indices(),[&](IndexSynapseInterval weight_synapse){
        std::uint32_t synapse_weights_done = 0u;
        if(first_weight_synapse_in_neuron){
          spike_weight_index = weight_synapse.starts();
          ++synapse_weights_done;
        }

        while(
          (synapse_weights_done < weight_synapse.interval_size())
          &&(input_synapse_index_offset < partial.index_synapse_number(inner_neuron_index))
        ){
          const std::uint32_t current_input_synapse_size = partial.inside_indices(input_synapse_index + input_synapse_index_offset).interval_size();
          std::uint32_t weights_able_to_do = std::min(
            (weight_synapse.interval_size() - synapse_weights_done), (current_input_synapse_size - input_offset_in_current_synapse)
          );

          /* decide input index start for this synapse */
          std::int32_t input_past_reach = partial.inside_indices(input_synapse_index + input_synapse_index_offset).reach_past_loops();
          std::int32_t input_index_start = partial.inside_indices(input_synapse_index + input_synapse_index_offset).starts();
          RFASSERT_LOG("InnerNeuron[{} / {}]: ", inner_neuron_index, partial.output_data().interval_size());
          RFASSERT_LOG("synapse_weights_done: {}/{}", synapse_weights_done, weight_synapse.interval_size());
          RFASSERT_LOG("input_synapse_index_offset: {}/{}", input_synapse_index_offset, partial.index_synapse_number(inner_neuron_index));
          RFASSERT_LOG("past: ", partial.inside_indices(input_synapse_index + input_synapse_index_offset).reach_past_loops());
          RFASSERT_LOG("current size: {}", current_input_synapse_size);
          RFASSERT_LOG("weights_able_to_do: {}", weights_able_to_do);
          RFASSERT_LOG("input_offset_in_current_synapse: {}", input_offset_in_current_synapse);
          RFASSERT_LOG("===========================================");

          bool index_points_to_input = false;
          if(SynapseIterator<>::is_index_input(input_index_start)){
            input_index_start = SynapseIterator<>::array_index_from_external_index(input_index_start - input_offset_in_current_synapse);
            input_past_reach = partial_input_synapses.reach_past_loops<InputSynapseInterval>(input_index_start);
            weights_able_to_do = std::min(
              weights_able_to_do, partial_input_synapses.interval_size_of(input_index_start)
            );
            input_index_start = partial_input_synapses[input_index_start];
            if(SynapseIterator<>::is_index_input(input_index_start)){
              input_index_start = SynapseIterator<>::array_index_from_external_index(input_index_start);
              index_points_to_input = true;
            }
          }else input_index_start = input_index_start + input_offset_in_current_synapse;

          /* decide input string for this synapse */
          std::function<std::string(std::string)> input_string;
          if(index_points_to_input){
            RFASSERT( 0 == input_past_reach );
            input_string = [input_index_start](std::string addition){
              return "inputs[input_start + " + addition + " + " + std::to_string(input_index_start) + "]";
            };
          }else if(0 != input_past_reach){
            input_string = [input_index_start, input_past_reach, past_reach_guard, &solution](std::string addition){
              return past_reach_guard(
                input_past_reach,
                "outputs[output_start + " + addition + " + " + std::to_string(input_index_start - static_cast<std::int32_t>(input_past_reach * solution.neuron_number())) + "]"
              );
            };
          }else input_string = [input_index_start](std::string addition){
            return "outputs[output_start + " + addition + " + " + std::to_string(input_index_start) + "]"; /* input doesn't reach to the past */
          };

          /* decide weight string for this synapse */
          std::function<std::string(std::string)> weight_string = [weight_table_offset, weight_synapse, synapse_weights_done](std::string addition){
            return "inputs[" + addition + " + " + std::to_string(weight_table_offset + weight_synapse.starts() + synapse_weights_done) + "]";
          };
          std::string input_lambda_correction = "";
          std::string inside_weight_start = "0";
          if(first_input_in_neuron){
            first_input_in_neuron = false;
            input_lambda_correction = "neuron_partial_result = (" + input_string("0") + " *" + weight_string("0") + " );";
            inside_weight_start = "1";
          }
          std::string neuron_input_operation = (
            "neuron_partial_result = "
            + InputFunction::get_kernel_function_for(
              partial.neuron_input_functions(inner_neuron_index),
              "neuron_partial_result", "(" + input_string("inside_weight_index") + " * " + weight_string("inside_weight_index") + ")"
            ) + ";"
          );
          std::string synapse_input_operation = (input_lambda_correction + R"(
            for(int inside_weight_index = ==inside_weight_start==; inside_weight_index < ==inputs_in_synapse==; ++inside_weight_index){
              ==neuron_input_operation==
            })"
          );
          synapse_input_operation = std::regex_replace(synapse_input_operation, std::regex("==inside_weight_start=="), inside_weight_start);
          synapse_input_operation = std::regex_replace(synapse_input_operation, std::regex("==inputs_in_synapse=="), std::to_string(weights_able_to_do));
          synapse_input_operation = std::regex_replace(synapse_input_operation, std::regex("==neuron_input_operation=="), neuron_input_operation);
          inner_neuron_operation += synapse_input_operation;
          if(0 < weights_able_to_do)
            first_weight_synapse_in_neuron = false;
          input_offset_in_current_synapse += weights_able_to_do;
          if(input_offset_in_current_synapse >= current_input_synapse_size){
            input_offset_in_current_synapse = 0;
            ++input_synapse_index_offset;
          }
          synapse_weights_done += weights_able_to_do;
        }/*while(there are still input synapses)*/
        RFASSERT(weight_synapse.interval_size() >= synapse_weights_done);

        /* operations for bias weights */
        if(input_synapse_index_offset >= partial.index_synapse_number(inner_neuron_index)){
          std::string input_lambda_correction = "";
          std::string inside_weight_start = "0";
          if(first_input_in_neuron){
            first_input_in_neuron = false;
            input_lambda_correction = (
              "neuron_partial_result = inputs["
              + std::to_string(weight_table_offset + weight_synapse.starts() + synapse_weights_done)
              + "];\n"
            );
            inside_weight_start = "1";
          }
          std::string neuron_bias_operation = (
            "neuron_partial_result = "
            + InputFunction::get_kernel_function_for(
                partial.neuron_input_functions(inner_neuron_index),
                "neuron_partial_result",
                "inputs[" + std::to_string(weight_table_offset + weight_synapse.starts() + synapse_weights_done) + " + inside_weight_index]"
            ) + ";"
          );
          std::string synapse_bias_operation = (input_lambda_correction + R"(
            for(int inside_weight_index = ==inside_weight_start==; inside_weight_index < ==biases_in_synapse==; ++inside_weight_index){
              ==neuron_bias_operation==
            })"
          );
          synapse_bias_operation = std::regex_replace(synapse_bias_operation, std::regex("==inside_weight_start=="), inside_weight_start);
          synapse_bias_operation = std::regex_replace(synapse_bias_operation, std::regex("==biases_in_synapse=="), std::to_string(weight_synapse.interval_size() - synapse_weights_done));
          synapse_bias_operation = std::regex_replace(synapse_bias_operation, std::regex("==neuron_bias_operation=="), neuron_bias_operation);
          inner_neuron_operation += synapse_bias_operation;
        }/*if(there are no more inputs to pair to the weights --> all inputs are biases)*/
        first_weight_synapse_in_neuron = false;
      }, weight_synapse_start, partial.weight_synapse_number(inner_neuron_index));

      const std::string neuron_index_str = std::to_string(partial.output_data().starts() + inner_neuron_index);
      inner_neuron_operation += "neuron_partial_result = " + transfer_function.get_kernel_function_for(
        partial.neuron_transfer_functions(inner_neuron_index),
        "neuron_partial_result"
      )+";\n";
      inner_neuron_operation += (
        "outputs[output_start + " + neuron_index_str + "] = (\n"
        + SpikeFunction::get_kernel_function_for(
          partial.neuron_spike_functions(inner_neuron_index),
          "neuron_partial_result"/* new_data */,
          past_reach_guard( 1/*past_reach*/, std::string("(outputs[output_start + " + neuron_index_str + " - neuron_array_size])"))/* previous_data */,
          "inputs[" + std::to_string(weight_table_offset + spike_weight_index) + "]"/*parameter*/
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
      if(NeuronInfo::is_feature_relevant_to_solution(feature.feature())){
          RafkoNetworkFeature::add_default_kernel_code_to(
            neuron_operations, feature, settings, solution,
            ""/*input_array*/, ""/*input_start_index*/, "outputs", "output_start", !feature_locals_declared
          );
          /*!Note: in solution relevant features, only the output array is used, so no need to add any input info */
          feature_locals_declared = true;
        }else if(NeuronInfo::is_feature_relevant_to_performance(feature.feature())){
          performance_feature_list.push_back(feature);
        }
      }
    }

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
    RafkoNetworkFeature::add_default_kernel_code_to(
      performance_operations, feature, settings, solution,
      "inputs"/*input_array*/, "1u"/*input_start_index:weight table start*/,
      "outputs"/*output_array*/, "output_sizes[0]"/*output_start_index: last output number stores the performance errors */,
      declare_locals
    );/*!Note: Disentanglement would require the the input to be of the Neuron array */

    if(declare_locals)
     already_declared_locals.insert(feature.feature());
  }
  source_base = std::regex_replace(source_base, std::regex("==name=="), name);
  source_base = std::regex_replace(source_base, std::regex("==neuron_array_size=="), std::to_string(solution.neuron_number()));
  source_base = std::regex_replace(source_base, std::regex("==weight_table_offset=="), std::to_string(weight_table_offset));
  source_base = std::regex_replace(source_base, std::regex("==network_memory_slots=="), std::to_string(solution.network_memory_length()));
  source_base = std::regex_replace(source_base, std::regex("==sequence_size=="), std::to_string(sequence_size));
  source_base = std::regex_replace(source_base, std::regex("==neuron_operations=="), neuron_operations);
  source_base = std::regex_replace(source_base, std::regex("==prefill_input_num=="), std::to_string(prefill_input_num));
  source_base = std::regex_replace(source_base, std::regex("==network_input_size=="), std::to_string(solution.network_input_size()));
  source_base = std::regex_replace(source_base, std::regex("==performance_operations=="), performance_operations);
  RFASSERT_LOG("Kernel code: {}", source_base);
  return source_base;
}
#endif/*(RAFKO_USES_OPENCL)*/


} /* namespace rafko_net */
