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

#define CATCH_CONFIG_RUNNER

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <cfloat>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_gym/models/rafko_cost.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/models/const_vector_subrange.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/models/transfer_function.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"

#include "test/test_utility.h"

int main( int argc, char* argv[] ) {
  int result = Catch::Session().run( argc, argv );
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}

namespace rafko_test {

void manual_2_neuron_partial_solution(rafko_net::PartialSolution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset){

  rafko_net::InputSynapseInterval temp_input_interval;
  rafko_net::IndexSynapseInterval temp_index_interval;

  /**###################################################################################################
   * Neuron global parameters in partial
   */
  temp_index_interval.set_starts(neuron_offset + 0u);
  temp_index_interval.set_interval_size(2);
  *partial_solution.mutable_output_data() = temp_index_interval;

  partial_solution.add_weight_table(double_literal(0.0)); /* spike function weight for first neuron */
  for(uint32 i = 0; i < number_of_inputs; ++i){
    partial_solution.add_weight_table(double_literal(1.0)); /* weight for the inputs coming to the first Neuron */
  } /* Every weight shall be modified in this example, so they'll all have thir own weight table entry */
  partial_solution.add_weight_table(double_literal(50.0)); /* first neuron bias value */
  partial_solution.add_weight_table(double_literal(0.0)); /* spike function weight for second neuron */
  partial_solution.add_weight_table(double_literal(1.0)); /* Weight for the first Neuron */
  partial_solution.add_weight_table(double_literal(10.0)); /* 2nd neuron bias value */

  /**###################################################################################################
   * The first neuron shall have the inputs
   */
  partial_solution.add_neuron_input_functions(rafko_net::input_function_add);
  partial_solution.add_neuron_transfer_functions(rafko_net::transfer_function_identity);

  /* inputs go to neuron1 */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights */
  temp_input_interval.set_starts(rafko_net::SynapseIterator<>::synapse_index_from_input_index(0)); /* Input index synapse starts at the beginning of the data */
  temp_input_interval.set_interval_size(number_of_inputs); /* Neuron 1 has an input index synapse of the inputs */
  *partial_solution.add_inside_indices() = temp_input_interval;

  partial_solution.add_weight_synapse_number(1u);
  temp_index_interval.set_starts(0u);
  temp_index_interval.set_interval_size(1u + number_of_inputs + 1u); /* Neuron 1 has the input weights, the bias and the spike function weight */
  *partial_solution.add_weight_indices() = temp_index_interval;

  /**###################################################################################################
   * The second Neuron shall only have the first neuron as input
   */
  partial_solution.add_neuron_input_functions(rafko_net::input_function_add);
  partial_solution.add_neuron_transfer_functions(rafko_net::transfer_function_identity);
  /* neuron1 goes to neuron2;  that is the output which isn't in the inside indexes */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights*/
  temp_input_interval.set_starts(0u); /* The input synapse starts at the 1st internal Neuron (index 0) */
  temp_input_interval.set_interval_size(1u); /* Neuron 2 has an input synapse of size 1 plus a bias*/
  *partial_solution.add_inside_indices() = temp_input_interval;

  partial_solution.add_weight_synapse_number(1u);
  temp_index_interval.set_starts(1u + number_of_inputs + 1u); /* spike weight 1 + number of inputs + bias1 */
  temp_index_interval.set_interval_size(1u + 1u + 1u); /* a spike function weight; a weight synapse of size 1 + a bias */
  *partial_solution.add_weight_indices() = temp_index_interval;
}

void manual_2_neuron_result(const std::vector<sdouble32>& partial_inputs, std::vector<sdouble32>& prev_neuron_output, const rafko_net::PartialSolution& partial_solution, uint32 neuron_offset){
  rafko_mainframe::RafkoSettings settings;
  rafko_net::TransferFunction trasfer_function(settings);

  /* Neuron 1 */
  sdouble32 neuron1_result = 0;
  for(uint32 weight_iterator = 1; weight_iterator <= partial_inputs.size(); ++weight_iterator){
    neuron1_result += (partial_inputs[weight_iterator - 1u] * partial_solution.weight_table(weight_iterator));
  }
  neuron1_result += partial_solution.weight_table(1u + partial_inputs.size()); /* spike weight 1 + inputs */
  neuron1_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(0),neuron1_result);
  prev_neuron_output[neuron_offset + 0] = prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(0)
   + neuron1_result * (double_literal(1.0) - partial_solution.weight_table(0));

  /* Neuron 2 */
  sdouble32 neuron2_result = (
    (prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(1u + partial_inputs.size() + 1u + 1u))
    + partial_solution.weight_table(1u + partial_inputs.size() + 1u + 1u + 1u)
  );

  neuron2_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(1),neuron2_result);
  prev_neuron_output[neuron_offset + 1] = (
    prev_neuron_output[neuron_offset + 1] * partial_solution.weight_table(1u + partial_inputs.size() + 1u)
    + ( neuron2_result * (double_literal(1.0) - partial_solution.weight_table(1u + partial_inputs.size() + 1u)) )
 );
}

void manaual_fully_connected_network_result(
  std::vector<sdouble32>& inputs, std::vector<sdouble32> previous_data, std::vector<sdouble32>& neuron_data,
  std::vector<uint32> layer_structure, rafko_net::RafkoNet network
){
  rafko_mainframe::RafkoSettings settings;
  rafko_net::TransferFunction trasfer_function(settings);

  uint32 neuron_number = 0;
  for(uint32 layer_iterator = 0; layer_iterator < layer_structure.size(); ++layer_iterator){ /* Go through all of the layers, count the number of Neurons */
    neuron_number += layer_structure[layer_iterator]; /* Sum the number of neurons accoding to the given layer structure */
  }
  REQUIRE(static_cast<sint32>(neuron_number) == network.neuron_array_size());
  if(0 == neuron_data.size())neuron_data = std::vector<sdouble32>(neuron_number);
  sdouble32 new_neuron_data = 0;
  sdouble32 neuron_input_value = 0;
  sdouble32 spike_function_weight;
  uint32 input_synapse_index = 0;
  uint32 input_index_offset = 0;
  bool first_weight_in_synapse;
  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    const rafko_net::Neuron& neuron = network.neuron_array(neuron_iterator);
    new_neuron_data = 0;
    input_synapse_index = 0;
    input_index_offset = 0;

    if(0 < previous_data.size())
      REQUIRE( neuron_data.size() == previous_data.size() );
    first_weight_in_synapse = true;
    rafko_net::SynapseIterator<>::iterate(neuron.input_weights(),[&](sint32 neuron_weight_index){
      if(true == first_weight_in_synapse){
        first_weight_in_synapse = false;
        spike_function_weight = network.weight_table(neuron_weight_index);
      }else{
        if(static_cast<sdouble32>(input_synapse_index) < neuron.input_indices_size()){ /* Only get input from the net if it's explicitly defined */
          REQUIRE( 1 >= neuron.input_indices(input_synapse_index).reach_past_loops() ); /* Only the last loop and the current can be handled in this test yet */
          if(rafko_net::SynapseIterator<>::is_index_input(neuron.input_indices(input_synapse_index).starts()))
            neuron_input_value = inputs[rafko_net::SynapseIterator<>::input_index_from_synapse_index(
              neuron.input_indices(input_synapse_index).starts() - input_index_offset
            )];
          else if(1 == neuron.input_indices(input_synapse_index).reach_past_loops())
            neuron_input_value = previous_data[ /* Neuron input is from network input 1 loop from the past */
              neuron.input_indices(input_synapse_index).starts() + input_index_offset
            ];
          else neuron_input_value = neuron_data[ /* Neuron input is from the current internal data of the network */
            neuron.input_indices(input_synapse_index).starts() + input_index_offset
          ];
          ++input_index_offset;
          if(neuron.input_indices(input_synapse_index).interval_size() <= input_index_offset){
            input_index_offset = 0;
            ++input_synapse_index;
          }
        }else{
          neuron_input_value = 1.0;
        }
        new_neuron_data += neuron_input_value * network.weight_table(neuron_weight_index);
      }
    }); /* For every weight in the Neuron sum the weigthed input*/
    /* apply transfer function */
    new_neuron_data = trasfer_function.get_value(neuron.transfer_function(),new_neuron_data);
    neuron_data[neuron_iterator] = ( /* Apply memory filter and save output to Neuron data */
      (neuron_data[neuron_iterator] * spike_function_weight)
      + new_neuron_data * (double_literal(1.0) - spike_function_weight)
    );
  } /* For every Neuron */
}

void check_if_the_same(const rafko_net::RafkoNet& net, const rafko_net::Solution& solution){
  uint32 input_synapse_offset;
  uint32 weight_synapse_offset;
  uint32 neuron_synapse_element_iterator;
  uint32 counted_inputs;
  uint32 expected_inputs;
  for(sint32 neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){ /* For the input Neurons */
    for(
      sint32 partial_solution_iterator = 0;
      partial_solution_iterator < solution.partial_solutions_size();
      ++partial_solution_iterator
    ){ /* Search trough the partial solutions, looking for the neuron_iterator'th Neuron */
      input_synapse_offset = 0; /* Since the Neurons are sharing their input synapses in a common array, an offset needs to be calculated */
      weight_synapse_offset = 0;

      /* Since Neurons take their inputs from the partial solution input, test iterates over it */
      rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> partial_input_iterator(solution.partial_solutions(partial_solution_iterator).input_data());
      const uint32 first_neuron_index_in_partial = solution.partial_solutions(partial_solution_iterator).output_data().starts();
      for( /* Skim through the inner neurons in the partial solution until the current one is found */
        uint32 i_neuron_iter = 0; i_neuron_iter < solution.partial_solutions(partial_solution_iterator).output_data().interval_size();++i_neuron_iter
      ){ /*!Note: i_neuron_iter == inner neuron iterator */
        if(neuron_iterator == static_cast<sint32>(first_neuron_index_in_partial + i_neuron_iter)){
          /* If the current neuron being checked is the one in the partial solution under i_neuron_iter */
          neuron_synapse_element_iterator = 0;

          /* Test iterates over the Neurons input weights, to see if they match with the weights of the Neurons inside the Network */
          rafko_net::SynapseIterator<> inner_neuron_weight_iterator(solution.partial_solutions(partial_solution_iterator).weight_indices());
          rafko_net::SynapseIterator<> neuron_weight_iterator(net.neuron_array(neuron_iterator).input_weights());

          /* Inner Neuron inputs point to indexes in the partial solution input ( when SynapseIterator<>::is_index_input is true ) */
          expected_inputs = 0;
          inner_neuron_weight_iterator.iterate([&](rafko_net::IndexSynapseInterval weight_synapse){
            expected_inputs += weight_synapse.interval_size();
          },[&](sint32 weight_index){
            REQUIRE( neuron_weight_iterator.size() > neuron_synapse_element_iterator );
            CHECK(
              solution.partial_solutions(partial_solution_iterator).weight_table(weight_index)
              == net.weight_table(neuron_weight_iterator[neuron_synapse_element_iterator])
            );
            ++neuron_synapse_element_iterator;
          },weight_synapse_offset,solution.partial_solutions(partial_solution_iterator).weight_synapse_number(i_neuron_iter));

          /* Test if all of the neurons inputs are are the same as the ones in the net */
          /* Test iterates over the inner neurons synapse to see if it matches the Neuron synapse */
          rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> inner_neuron_input_iterator(solution.partial_solutions(partial_solution_iterator).inside_indices());
          rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> neuron_input_iterator(net.neuron_array(neuron_iterator).input_indices());

          /* Neuron inputs point to indexes in the partial solution input ( when SynapseIterator<>::is_index_input s true ) */
          neuron_synapse_element_iterator = 0;
          counted_inputs = 0;
          uint32 current_reachback;
          inner_neuron_input_iterator.iterate([&](rafko_net::InputSynapseInterval input_synapse){
            current_reachback = input_synapse.reach_past_loops();
          },[&](sint32 input_index){
            REQUIRE( neuron_input_iterator.size() > neuron_synapse_element_iterator );
            if(!rafko_net::SynapseIterator<>::is_index_input(input_index)){ /* Inner neuron takes its input internally */
              CHECK( 0 == current_reachback ); /* Internal inputs should always be taken from the current loop */
              REQUIRE(
                static_cast<sint32>(first_neuron_index_in_partial + input_index)
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
            }else{ /* Inner Neuron takes its input from the partial solution input */
              REQUIRE( /* Input indices match */
                partial_input_iterator[rafko_net::SynapseIterator<>::input_index_from_synapse_index(input_index)]
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
             REQUIRE( /* The time the neuron takes its input also match */
                partial_input_iterator.synapse_under(rafko_net::SynapseIterator<>::input_index_from_synapse_index(input_index)).reach_past_loops()
                == neuron_input_iterator.synapse_under(neuron_synapse_element_iterator).reach_past_loops()
              );
            }
            ++neuron_synapse_element_iterator;
            ++counted_inputs;
          },input_synapse_offset,solution.partial_solutions(partial_solution_iterator).index_synapse_number(i_neuron_iter));
          REQUIRE( neuron_input_iterator.size() == counted_inputs );
          goto Neuron_found_in_partial;
        }else{ /* neuron_iterator is not under i_neuron_iter in the partial solution.. adjust synapse offsets */
          input_synapse_offset += solution.partial_solutions(partial_solution_iterator).index_synapse_number(i_neuron_iter);
          weight_synapse_offset += solution.partial_solutions(partial_solution_iterator).weight_synapse_number(i_neuron_iter);
        }
      } /* Inner Neuron loop*/
    } /* Partial solution loop */
    Neuron_found_in_partial:;
  } /*(uint32 neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator)*/
}

void print_weights(const rafko_net::RafkoNet& net, const rafko_net::Solution& solution){
  std::cout << "net("<< net.weight_table_size() << " weights):";
  for(sint32 weight_index = 0; weight_index < net.weight_table_size(); ++weight_index){
    std::cout << "["<< net.weight_table(weight_index) <<"]";
  }
  std::cout << std::endl << "ptls( "<< solution.partial_solutions_size() << " partials):";
  for(sint32 partial_index = 0; partial_index < solution.partial_solutions_size(); ++partial_index){
    for(sint32 weight_index = 0; weight_index < solution.partial_solutions(partial_index).weight_table_size(); ++weight_index){
      std::cout << "["<< solution.partial_solutions(partial_index).weight_table(weight_index) <<"]";
    }
    std::cout << std::endl;
  }
}

void print_training_sample(
  uint32 sample_sequence_index, rafko_gym::RafkoDatasetWrapper& data_set,
  const rafko_net::RafkoNet& net, const rafko_mainframe::RafkoSettings& settings
){
  std::unique_ptr<rafko_net::Solution> solution = rafko_net::SolutionBuilder(settings).build(net);
  std::unique_ptr<rafko_net::SolutionSolver> sample_solver(
    rafko_net::SolutionSolver::Builder(*solution, settings).build()
  );
  std::vector<sdouble32> neuron_data(data_set.get_sequence_size());
  uint32 raw_label_index = sample_sequence_index;
  uint32 raw_inputs_index = raw_label_index * (data_set.get_sequence_size() + data_set.get_prefill_inputs_number());
  raw_label_index *= data_set.get_sequence_size();

  std::cout.precision(2);
  std::cout << std::endl << "Training sample["<< sample_sequence_index <<"]:" << std::endl;

  /* Prefill neural network */
  for(uint32 prefill_iterator = 0; prefill_iterator < data_set.get_prefill_inputs_number(); ++prefill_iterator){
    (void)sample_solver->solve(data_set.get_input_sample(raw_inputs_index), (0 == prefill_iterator), 0);
    ++raw_inputs_index;
  } /* The first few labels are there to set an initial state to the network */


  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "["<< data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[0] <<"]";
  }
  std::cout << std::endl;
  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "["<< data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[1] <<"]";
  }
  std::cout << std::endl;
  std::cout << "--------------expected:" << std::endl;
  std::cout.precision(2);
  rafko_utilities::DataRingbuffer output_data_copy(0,0);
  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "\t\t["<< data_set.get_label_sample(raw_label_index)[0] <<"]";
    rafko_utilities::ConstVectorSubrange output_data = sample_solver->solve(
      data_set.get_input_sample(raw_inputs_index),
      ( (0u == data_set.get_prefill_inputs_number())&&(0u == j) ),
      /* thread index */0u
    );
    neuron_data[j] = output_data.back();
    output_data_copy = sample_solver->get_memory(0);
    ++raw_label_index;
    ++raw_inputs_index;
  }
  std::cout << std::endl;
  std::cout << "------<>------actual:" << std::endl;

  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "\t\t["<< neuron_data[j] <<"]";
  }
  std::cout << std::endl;

  std::cout << "==============" << std::endl;
  std::cout << "Neural memory for current sequence: " << std::endl;
  for(const std::vector<sdouble32>& vector : output_data_copy.get_whole_buffer()){
    for(const sdouble32& element : vector) std::cout << "[" << element << "]";
    std::cout << std::endl;
  }
  std::cout << "weights: " << std::endl;
  for(int i = 0; i < net.weight_table_size(); ++i){
    std::cout << "[" << net.weight_table(i) << "]";
  }
  std::cout << std::endl;
}

std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> create_addition_dataset(uint32 number_of_samples){

  std::vector<std::vector<sdouble32>> net_inputs(number_of_samples);
  std::vector<std::vector<sdouble32>> net_labels(number_of_samples);

  srand(time(nullptr));
  sdouble32 max_x = DBL_MIN;
  sdouble32 max_y = DBL_MIN;
  for(uint32 i = 0;i < number_of_samples;++i){
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    net_inputs[i].push_back(static_cast<sdouble32>(rand()%100));
    if(net_inputs[i][0] > max_x)max_x = net_inputs[i][0];
    if(net_inputs[i][1] > max_y)max_y = net_inputs[i][1];
  }

  for(uint32 i = 0;i < number_of_samples;++i){ /* Normalize the inputs */
    net_inputs[i][0] /= max_x;
    net_inputs[i][1] /= max_y;
    net_labels[i].push_back(net_inputs[i][0] + net_inputs[i][1]);
  }
  return std::make_pair(net_inputs,net_labels);
}

std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> create_sequenced_addition_dataset(uint32 number_of_samples, uint32 sequence_size){
  uint32 carry_bit;
  std::vector<std::vector<sdouble32>> net_inputs(sequence_size * number_of_samples);
  std::vector<std::vector<sdouble32>> net_labels(sequence_size * number_of_samples);

  srand(time(nullptr));
  for(uint32 i = 0;i < number_of_samples;++i){
    carry_bit = 0;
    for(uint32 j = 0;j <sequence_size;++j){ /* Add testing and training sequences randomly */
      net_inputs[(sequence_size * i) + j] = std::vector<sdouble32>(2);
      net_labels[(sequence_size * i) + j] = std::vector<sdouble32>(1);
      net_inputs[(sequence_size * i) + j][0] = static_cast<sdouble32>(rand()%2);
      net_inputs[(sequence_size * i) + j][1] = static_cast<sdouble32>(rand()%2);

      net_labels[(sequence_size * i) + j][0] =
        net_inputs[(sequence_size * i) + j][0]
        + net_inputs[(sequence_size * i) + j][1]
        + carry_bit;
      if(1 < net_labels[(sequence_size * i) + j][0]){
        net_labels[(sequence_size * i) + j][0] = 1;
        carry_bit = 1;
      }else{
        carry_bit = 0;
      }
    }
  }
  return std::make_pair(net_inputs,net_labels);
}


TEST_CASE("Testing whether binary addition can be solved with a manual program","[meta]"){
  uint32 sequence_size = 4;
  uint32 number_of_samples = 10;
  std::pair<std::vector<std::vector<sdouble32>>,std::vector<std::vector<sdouble32>>> dataset = create_sequenced_addition_dataset(number_of_samples, sequence_size);
  std::vector<std::vector<sdouble32>>& inputs = std::get<0>(dataset);
  std::vector<std::vector<sdouble32>>& labels = std::get<1>(dataset);

  for(uint32 sample_iterator = 0; sample_iterator < number_of_samples; ++sample_iterator){
    sdouble32 carry_bit = 0;
    for(uint32 sequence_iterator = 0; sequence_iterator < sequence_size; ++sequence_iterator){
      sdouble32 result = (
        inputs[(sample_iterator * sequence_size) + sequence_iterator][0]
        + inputs[(sample_iterator * sequence_size) + sequence_iterator][1]
        + carry_bit
      );
      carry_bit = (double_literal(1.0) < result)?double_literal(1.0):double_literal(0.0);
      result = std::min(double_literal(1.0), result);
      REQUIRE( std::min(double_literal(1.0), result) == labels[(sample_iterator * sequence_size) + sequence_iterator][0] );
    }
  }
}

rafko_net::RafkoNet* generate_random_net_with_softmax_features(uint32 input_size, rafko_mainframe::RafkoSettings& settings){
  std::vector<uint32> net_structure;
  while((rand()%10 < 9)||(4 > net_structure.size()))
    net_structure.push_back(static_cast<uint32>(rand()%5) + 1u);

  uint8 num_of_features = rand()%(net_structure.size()/2) + 1u;
  rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
    .input_size(input_size)
    .expected_input_range(double_literal(5.0));

  uint8 layer_of_feature_index = 0;
  uint32 layer_start_index = 0;
  uint8 feature_index;
  for(feature_index = 0u; feature_index < num_of_features; feature_index++){
    if(layer_of_feature_index >= net_structure.size())break;
    uint8 layer_diff = 1u + ((rand()%(net_structure.size() - layer_of_feature_index)) / 2);
    for(uint8 i = 0; i < layer_diff; ++i){
      layer_start_index += net_structure[layer_of_feature_index + i];
    }
    layer_of_feature_index += layer_diff;
    builder.add_feature_to_layer(layer_of_feature_index, rafko_net::neuron_group_feature_softmax);
  }

  return builder.dense_layers(net_structure);
}

std::unique_ptr<rafko_gym::DataSet> create_dataset(
  uint32 input_size, uint32 feature_size,
  uint32 sample_number, uint32 sequence_size, uint32 prefill_size,
  sdouble32 expected_label, sdouble32 label_delta_per_feature
){
  std::unique_ptr<rafko_gym::DataSet> dataset = std::make_unique<rafko_gym::DataSet>();
  dataset->set_input_size(input_size);
  dataset->set_feature_size(feature_size);
  dataset->set_sequence_size(sequence_size);
  /*!Note: Input values should be irrelevant in test sets like this */
  sdouble32 label_value = expected_label;
  for(uint32 sequence_index = 0; sequence_index < (sample_number); ++sequence_index){
    for(uint32 prefill_index = 0; prefill_index < prefill_size; ++prefill_index){
      for(uint32 input_index = 0; input_index < input_size; ++input_index) dataset->add_inputs(5 - prefill_index);
    }
    for(uint32 raw_label_index = 0; raw_label_index < sequence_size; ++raw_label_index){
      for(uint32 input_index = 0; input_index < input_size; ++input_index) dataset->add_inputs(sequence_index + input_index);
      for(uint32 feature_index = 0; feature_index < feature_size; ++feature_index){
        dataset->add_labels(label_value);
        label_value += label_delta_per_feature;
      }
    }
  }/*for(every sequence)*/
  return dataset;
}

} /* namsepace rafko_test */
