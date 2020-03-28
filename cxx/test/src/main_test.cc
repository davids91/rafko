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

#include "test/catch.hpp"

#include "test/test_mockups.h"
#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/transfer_function.h"
#include "services/synapse_iterator.h"

int main( int argc, char* argv[] ) {

  Catch::cout() << "Catch version "
  << CATCH_VERSION_MAJOR << "." << CATCH_VERSION_MINOR << "."
  << CATCH_VERSION_PATCH <<std::endl;

  int result = Catch::Session().run( argc, argv );

  google::protobuf::ShutdownProtobufLibrary();

  return result;
}

namespace sparse_net_library_test{

using sparse_net_library::sint32;
using sparse_net_library::sdouble32;
using sparse_net_library::Transfer_function;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::Input_synapse_interval;
using sparse_net_library::Index_synapse_interval;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Neuron;

void manual_2_neuron_partial_solution(Partial_solution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset){

  Input_synapse_interval temp_input_interval;
  Index_synapse_interval temp_index_interval;

  /**###################################################################################################
   * Neuron global parameters in partial
   */
  partial_solution.set_internal_neuron_number(2);
  temp_index_interval.set_starts(neuron_offset + 0u);
  temp_index_interval.set_interval_size(2);
  *partial_solution.add_output_data() = temp_index_interval;

  for(uint32 i = 0; i < number_of_inputs; ++i){
    partial_solution.add_weight_table(double_literal(1.0)); /* weight for the inputs coming to the first Neuron */
  } /* Every weight shall be modified in this example, so they'll all have thir own weight table entry */
  partial_solution.add_weight_table(double_literal(50.0)); /* a bias value */
  partial_solution.add_weight_table(double_literal(0.0)); /* a memory ratio value */
  partial_solution.add_weight_table(double_literal(1.0)); /* Weight for the first Neuron */
  partial_solution.add_weight_table(double_literal(10.0)); /* a bias value */
  partial_solution.add_weight_table(double_literal(0.0)); /* a memory ratio value */

  /**###################################################################################################
   * The first neuron shall have the inputs
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_filter_index(
    number_of_inputs               + 1u
  ); /* input weights + first bias + first index */

  /* inputs go to neuron1 */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights */
  temp_input_interval.set_starts(Synapse_iterator<>::synapse_index_from_input_index(0)); /* Input index synapse starts at the beginning of the data */
  temp_input_interval.set_interval_size(number_of_inputs); /* Neuron 1 has an input index synapse of the inputs */
  *partial_solution.add_inside_indices() = temp_input_interval;

  partial_solution.add_weight_synapse_number(1u);
  temp_index_interval.set_starts(0u);
  temp_index_interval.set_interval_size(number_of_inputs + 1); /* Neuron 1 has the inputs in its only weight synapse */
  *partial_solution.add_weight_indices() = temp_index_interval;

  /**###################################################################################################
   * The second Neuron shall only have the first neuron as input
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_filter_index(
    number_of_inputs          + 1u                      + 1u                  + 1u          + 1u
  ); /* input weights + bias1 + first memory ratio value + first neuron weight + second bias + after the previous index */

  /* neuron1 goes to neuron2;  that is the output which isn't in the inside indexes */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights*/
  temp_input_interval.set_starts(0u); /* The input synapse starts at the 1st internal Neuron (index 0) */
  temp_input_interval.set_interval_size(1u); /* Neuron 2 has an input synapse of size 1 plus a bias*/
  *partial_solution.add_inside_indices() = temp_input_interval;
  partial_solution.add_weight_synapse_number(1u);
  temp_index_interval.set_starts(
    number_of_inputs             + 1u            + 1u
  ); /* number of inputs + bias1 + memory_ratio1 + index start*/
  temp_index_interval.set_interval_size(2); /* Neuron 2 has a an weight synapse of size 1 + a bias*/
  *partial_solution.add_weight_indices() = temp_index_interval;
}

void manual_2_neuron_result(const vector<sdouble32>& partial_inputs, vector<sdouble32>& prev_neuron_output, const Partial_solution& partial_solution, uint32 neuron_offset){
  Transfer_function trasfer_function;

  /* Neuron 1 = transfer_function( ( input0 * weight0 + input1 * weight1 ... inputN * weightN ) + bias0 )*/
  sdouble32 neuron1_result = 0;
  for(uint32 weight_iterator = 0; weight_iterator < partial_inputs.size(); ++weight_iterator){
    neuron1_result += (partial_inputs[weight_iterator] * partial_solution.weight_table(weight_iterator));
  }
  neuron1_result += partial_solution.weight_table(partial_inputs.size());
  neuron1_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(0),neuron1_result);
  prev_neuron_output[neuron_offset + 0] = prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(partial_solution.memory_filter_index(0))
   + neuron1_result * (double_literal(1.0) - partial_solution.weight_table(partial_solution.memory_filter_index(0)));

  /* Neuron 2 = transfer_function( (Neuron1 * weight[inputs + 1]) + bias1 ) */
  sdouble32 neuron2_result = (prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(partial_inputs.size() + 1u + 1u))
   + partial_solution.weight_table(partial_inputs.size() + 1u + 1u + 1u);

  neuron2_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(1),neuron2_result);
  prev_neuron_output[neuron_offset + 1] = prev_neuron_output[neuron_offset + 1] * partial_solution.weight_table(partial_solution.memory_filter_index(1))
   + neuron2_result * (double_literal(1.0) - partial_solution.weight_table(partial_solution.memory_filter_index(1)));
}

void manaual_fully_connected_network_result(vector<sdouble32> inputs, vector<sdouble32>& neuron_data,
    vector<uint32> layer_structure, SparseNet network){
  Transfer_function trasfer_function;
  uint32 neuron_number = 0;
  for(uint32 layer_iterator = 0; layer_iterator < layer_structure.size(); ++layer_iterator){ /* Go through all of the layers, count the number of Neurons */
    neuron_number += layer_structure[layer_iterator]; /* Sum the number of neurons accoding to the given layer structure */
  }
  if(static_cast<int>(neuron_number) != network.neuron_array_size())throw "Given Network Structure doesn't fit Network Neuron number!";
  if(0 == neuron_data.size())neuron_data = vector<sdouble32>(neuron_number);
  sdouble32 new_neuron_data = 0;
  sdouble32 neuron_input_value = 0;
  uint32 input_synapse_index = 0;
  uint32 input_index_offset = 0;
  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    const Neuron& neuron = network.neuron_array(neuron_iterator);
    new_neuron_data = 0;
    input_synapse_index = 0;
    input_index_offset = 0;

    Synapse_iterator<>::iterate(neuron.input_weights(),[&](int neuron_weight_index){
      if(static_cast<sdouble32>(input_synapse_index) < neuron.input_indices_size()){ /* Only get input from the net if it's explicitly defined */
        if(Synapse_iterator<>::is_index_input(neuron.input_indices(input_synapse_index).starts()))
          neuron_input_value = inputs[Synapse_iterator<>::input_index_from_synapse_index(
            neuron.input_indices(input_synapse_index).starts() - input_index_offset
          )];
        else neuron_input_value = neuron_data[
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
    }); /* For every weight in the Neuron sum the weigthed input*/
    /* apply transfer function */
    new_neuron_data = trasfer_function.get_value(neuron.transfer_function_idx(),new_neuron_data);
    neuron_data[neuron_iterator] = /* Apply memory filter and save output to Neuron data */
      neuron_data[neuron_iterator] * (network.weight_table(neuron.memory_filter_idx()))
      + new_neuron_data * (double_literal(1.0) - network.weight_table(neuron.memory_filter_idx()));
  } /* For every Neuron */
}

void check_if_the_same(SparseNet& net, Solution& solution){
  uint32 input_synapse_offset;
  uint32 weight_synapse_offset;
  uint32 neuron_synapse_element_iterator;
  for(sint32 neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){ /* For the input Neurons */
    for(
      sint32 partial_solution_iterator = 0;
      partial_solution_iterator < solution.partial_solutions_size();
      ++partial_solution_iterator
    ){ /* Search trough the partial solutions, looking for the neuron_iterator'th Neuron */
      input_synapse_offset = 0; /* Since the Neurons are sharing their input synapses in a common array, an offset needs to be calculated */
      weight_synapse_offset = 0;

      /* Since Neurons take their inputs from the partial solution input, test iterates over it */
      Synapse_iterator<Input_synapse_interval> partial_input_iterator(solution.partial_solutions(partial_solution_iterator).input_data());
      Synapse_iterator<> output_neurons(solution.partial_solutions(partial_solution_iterator).output_data());
      for( /* Skim through the inner neurons in the partial solution until the current one if found */
        uint32 inner_neuron_iterator = 0;
        inner_neuron_iterator < solution.partial_solutions(partial_solution_iterator).internal_neuron_number();
        ++inner_neuron_iterator
      ){
        if(neuron_iterator == output_neurons[inner_neuron_iterator]){
          /* If the current neuron being checked is the one in the partial solution under inner_neuron_iterator */
          neuron_synapse_element_iterator = 0;
          /* Test iterates over the Neurons input weights, to see if they match with the wights in the Network */
          Synapse_iterator<> inner_neuron_weight_iterator(solution.partial_solutions(partial_solution_iterator).weight_indices());
          Synapse_iterator<> neuron_weight_iterator(net.neuron_array(neuron_iterator).input_weights());
          inner_neuron_weight_iterator.iterate([&](sint32 input_index){ /* Inner Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator<>::is_index_input is true ) */
            REQUIRE( neuron_weight_iterator.size() > neuron_synapse_element_iterator );
            CHECK(
              solution.partial_solutions(partial_solution_iterator).weight_table(input_index)
              == net.weight_table(neuron_weight_iterator[neuron_synapse_element_iterator])
            );
            ++neuron_synapse_element_iterator;
          },weight_synapse_offset,solution.partial_solutions(partial_solution_iterator).weight_synapse_number(inner_neuron_iterator));

          /* Test if all of the neurons inputs are are the same as the ones in the net */
          neuron_synapse_element_iterator = 0;
          /* Test iterates over the inner neurons synapse to see if it matches the Neuron synapse */
          Synapse_iterator<Input_synapse_interval> inner_neuron_input_iterator(solution.partial_solutions(partial_solution_iterator).inside_indices());
          Synapse_iterator<Input_synapse_interval> neuron_input_iterator(net.neuron_array(neuron_iterator).input_indices());
          inner_neuron_input_iterator.iterate([&](sint32 input_index){ /* Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator<>::is_index_input s true ) */
            REQUIRE( neuron_input_iterator.size() > neuron_synapse_element_iterator );
            if(!Synapse_iterator<>::is_index_input(input_index)){ /* Inner neuron takes its input internally */
              CHECK(output_neurons[input_index] == neuron_input_iterator[neuron_synapse_element_iterator]);
            }else{ /* Inner Neuron takes its input from the partial solution input */
              CHECK(
                partial_input_iterator[Synapse_iterator<>::input_index_from_synapse_index(input_index)]
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
            }
            ++neuron_synapse_element_iterator;
          },input_synapse_offset,solution.partial_solutions(partial_solution_iterator).index_synapse_number(inner_neuron_iterator));
          goto Neuron_found_in_partial;
        }else{ /* neuron_iterator is not under inner_neuron_iterator in the partial solutio.. adjust synapse offsets */
          input_synapse_offset += solution.partial_solutions(partial_solution_iterator).index_synapse_number(inner_neuron_iterator);
          weight_synapse_offset += solution.partial_solutions(partial_solution_iterator).weight_synapse_number(inner_neuron_iterator);
        }
      } /* Inner Neuron loop*/
    } /* Partial solution loop */
    Neuron_found_in_partial:
    input_synapse_offset = 0; /* Dummy statement so that accursed goto works with the above label. Don't use GOTO kids! ..unless you absolutely have to! */
  } /*(uint32 neuron_iterator = 0; neuron_iterator < net.neuron_array_size(); ++neuron_iterator)*/
}


void print_weights(SparseNet& net, Solution& solution){
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

} /* namsepace sparse_net_library_test */
