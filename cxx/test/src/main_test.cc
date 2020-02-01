#define CATCH_CONFIG_MAIN  /* This tells Catch to provide a main() - only do this in one cpp file */

#include "test/catch.hpp"

#include "test/test_mockups.h"
#include "models/transfer_function.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library_test{

using sparse_net_library::sdouble32;
using sparse_net_library::Transfer_function;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Synapse_interval;
using sparse_net_library::Neuron;

void manual_2_neuron_partial_solution(Partial_solution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset){

  Synapse_interval temp_synapse_interval;

  /**###################################################################################################
   * Neuron global parameters in partial
   */
  partial_solution.set_internal_neuron_number(2);
  temp_synapse_interval.set_starts(neuron_offset + 0u);
  temp_synapse_interval.set_interval_size(2);
  *partial_solution.add_output_data() = temp_synapse_interval;

  for(uint32 i = 0; i < number_of_inputs; ++i){
    partial_solution.add_weight_table(1.0); /* weight for the inputs coming to the first Neuron */
  } /* Every weight shall be modified in this example, so they'll all have thir own weight table entry */
  partial_solution.add_weight_table(1.0); /* Weight for the first Neuron */
  partial_solution.add_weight_table(0.0); /* Memory ratios are also stored here */
  partial_solution.add_weight_table(0.0);
  partial_solution.add_weight_table(50.0); /* Biases are also stored here */
  partial_solution.add_weight_table(10.0);

  /* Add internal Neuron IDs */
  partial_solution.add_actual_index(neuron_offset + 0u); /* Really doesn't matter that much in this testcase */
  partial_solution.add_actual_index(neuron_offset + 1u); /* It will matter only when multiple partial partial_solutions are joind together */

  /**###################################################################################################
   * The first neuron shall have the inputs
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_filter_index(number_of_inputs + 1); /* input weights + first neuron weight + first index */
  partial_solution.add_bias_index(number_of_inputs + 2 + 1); /* input weights + first neuron weight + memory filters + first index*/

  /* inputs go to neuron1 */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights */
  temp_synapse_interval.set_starts(Synapse_iterator::synapse_index_from_input_index(0)); /* Input index synapse starts at the beginning of the data */
  temp_synapse_interval.set_interval_size(number_of_inputs); /* Neuron 1 has an input index synapse of the inputs */
  *partial_solution.add_inside_indices() = temp_synapse_interval;

  partial_solution.add_weight_synapse_number(1u);
  temp_synapse_interval.set_starts(0u);
  temp_synapse_interval.set_interval_size(number_of_inputs); /* Neuron 1 has the inputs in its only weight synapse */
  *partial_solution.add_weight_indices() = temp_synapse_interval;

  /**###################################################################################################
   * The second Neuron shall only have the first neuron as input
   */
  partial_solution.add_neuron_transfer_functions(TRANSFER_FUNCTION_IDENTITY);
  partial_solution.add_memory_filter_index(number_of_inputs + 2u); /* input weights + first neuron weight + second index */
  partial_solution.add_bias_index(number_of_inputs + 2u + 2u); /* input weights + first neuron weight + memory filters + second index*/

  /* neuron1 goes to neuron2;  that is the output which isn't in the inside indexes */
  partial_solution.add_index_synapse_number(1u); /* 1 synapse for indexes and 1 for weights*/
  temp_synapse_interval.set_starts(0u); /* The input synapse starts at the 1st internal Neuron (after the inputs) */
  temp_synapse_interval.set_interval_size(1); /* Neuron 2 has an input synapse of size 1*/
  *partial_solution.add_inside_indices() = temp_synapse_interval;
  partial_solution.add_weight_synapse_number(1u);
  temp_synapse_interval.set_starts(number_of_inputs); /* The weight synapse starts after the input weights in the weight table */
  temp_synapse_interval.set_interval_size(1); /* Neuron 2 has a an weight synapse of size 1 */
  *partial_solution.add_weight_indices() = temp_synapse_interval;

}

void manual_2_neuron_result(const vector<sdouble32>& partial_inputs, vector<sdouble32>& prev_neuron_output, const Partial_solution& partial_solution, uint32 neuron_offset){
  Transfer_function trasfer_function;

  /* Neuron 1 = transfer_function( ( input0 * weight0 + input1 * weight1 ... inputN * weightN ) + bias0 )*/
  sdouble32 neuron1_result = 0;
  for(uint32 weight_iterator = 0; weight_iterator < partial_inputs.size(); ++weight_iterator){
    neuron1_result += (partial_inputs[weight_iterator] * partial_solution.weight_table(weight_iterator));
  }

  neuron1_result += partial_solution.weight_table(partial_solution.bias_index(0));
  neuron1_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(0),neuron1_result);
  prev_neuron_output[neuron_offset + 0] = prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(partial_solution.memory_filter_index(0))
   + neuron1_result * (1.0 - partial_solution.weight_table(partial_solution.memory_filter_index(0)));

  /* Neuron 2 = transfer_function( (Neuron1 * weight[inputs + 1]) + bias1 ) */
  sdouble32 neuron2_result = (prev_neuron_output[neuron_offset + 0] * partial_solution.weight_table(partial_inputs.size()))
   + partial_solution.weight_table(partial_solution.bias_index(1));

  neuron2_result = trasfer_function.get_value(partial_solution.neuron_transfer_functions(1),neuron2_result);
  prev_neuron_output[neuron_offset + 1] = prev_neuron_output[neuron_offset + 1] * partial_solution.weight_table(partial_solution.memory_filter_index(1))
   + neuron2_result * (1.0 - partial_solution.weight_table(partial_solution.memory_filter_index(1)));
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
  uint32 weight_synapse_index = 0;
  uint32 weight_index = 0;

  for(uint32 neuron_iterator = 0; neuron_iterator < neuron_number; ++neuron_iterator){
    const Neuron& neuron = network.neuron_array(neuron_iterator);
    new_neuron_data = 0;
    weight_synapse_index = 0;
    weight_index = 0;

    Synapse_iterator neuron_input_iterator(neuron.input_indices());
    neuron_input_iterator.iterate([&](int neuron_input_index){
      if(Synapse_iterator::is_index_input(neuron_input_index)){
        neuron_input_value = inputs[Synapse_iterator::input_index_from_synapse_index(neuron_input_index)];
      }else{
        if(neuron_input_index > static_cast<int>(neuron_iterator)) throw "Neural Network contains input indexes not compatible with a Fully connected Neural Network";
        neuron_input_value = neuron_data[neuron_input_index];
      }
      if(neuron.input_weights_size() <= static_cast<int>(weight_synapse_index)) throw "Neural Network contains more inputs, than weights!";
      new_neuron_data += (neuron_input_value * network.weight_table(neuron.input_weights(weight_synapse_index).starts() + weight_index));
      ++weight_index;
      if(neuron.input_weights(weight_synapse_index).interval_size() <= weight_index){
        weight_index = 0;
        ++weight_synapse_index;
      }
    }); /* For every input in the Neuron sum the weigthed input*/
    new_neuron_data += network.weight_table(neuron.bias_idx()); /* add bias */
    new_neuron_data = trasfer_function.get_value(neuron.transfer_function_idx(),new_neuron_data); /* apply transfer function */
    neuron_data[neuron_iterator] = /* Apply memory filter and save output to Neuron data */
      neuron_data[neuron_iterator] * (network.weight_table(neuron.memory_filter_idx()))
      + new_neuron_data * (1.0 - network.weight_table(neuron.memory_filter_idx()));
  } /* For every Neuron */
}


} /* namsepace sparse_net_library_test */
