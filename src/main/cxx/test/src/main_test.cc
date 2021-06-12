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
#include "test/test_utility.h"

#include <deque>

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "rafko_mainframe/models/service_context.h"
#include "sparse_net_library/models/transfer_function.h"
#include "sparse_net_library/models/data_ringbuffer.h"
#include "sparse_net_library/models/data_aggregate.h"
#include "sparse_net_library/services/synapse_iterator.h"
#include "sparse_net_library/services/solution_builder.h"
#include "sparse_net_library/services/solution_solver.h"

int main( int argc, char* argv[] ) {
  int result = Catch::Session().run( argc, argv );
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}

namespace sparse_net_library_test {

using std::deque;

using sparse_net_library::DataRingbuffer;
using sparse_net_library::Data_aggregate;
using sparse_net_library::Transfer_function;
using sparse_net_library::TRANSFER_FUNCTION_IDENTITY;
using sparse_net_library::cost_functions;
using sparse_net_library::Input_synapse_interval;
using sparse_net_library::Index_synapse_interval;
using sparse_net_library::Solution_builder;
using sparse_net_library::Solution_solver;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Neuron;
using rafko_mainframe::Service_context;

void manual_2_neuron_partial_solution(Partial_solution& partial_solution, uint32 number_of_inputs, uint32 neuron_offset){

  Input_synapse_interval temp_input_interval;
  Index_synapse_interval temp_index_interval;

  /**###################################################################################################
   * Neuron global parameters in partial
   */
  partial_solution.set_internal_neuron_number(2);
  temp_index_interval.set_starts(neuron_offset + 0u);
  temp_index_interval.set_interval_size(2);
  *partial_solution.mutable_output_data() = temp_index_interval;

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
  Service_context service_context;
  Transfer_function trasfer_function(service_context);

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

void manaual_fully_connected_network_result(
  vector<sdouble32>& inputs, vector<sdouble32> previous_data, vector<sdouble32>& neuron_data,
  vector<uint32> layer_structure, SparseNet network
){
  Service_context service_context;
  Transfer_function trasfer_function(service_context);

  uint32 neuron_number = 0;
  for(uint32 layer_iterator = 0; layer_iterator < layer_structure.size(); ++layer_iterator){ /* Go through all of the layers, count the number of Neurons */
    neuron_number += layer_structure[layer_iterator]; /* Sum the number of neurons accoding to the given layer structure */
  }
  REQUIRE(static_cast<sint32>(neuron_number) == network.neuron_array_size());
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

    if(0 < previous_data.size())
      REQUIRE( neuron_data.size() == previous_data.size() );
    Synapse_iterator<>::iterate(neuron.input_weights(),[&](Index_synapse_interval weight_synapse, sint32 neuron_weight_index){
      if(static_cast<sdouble32>(input_synapse_index) < neuron.input_indices_size()){ /* Only get input from the net if it's explicitly defined */
        REQUIRE( 1 >= neuron.input_indices(input_synapse_index).reach_past_loops() ); /* Only the last loop and the current can be handled in this test yet */
        if(Synapse_iterator<>::is_index_input(neuron.input_indices(input_synapse_index).starts()))
          neuron_input_value = inputs[Synapse_iterator<>::input_index_from_synapse_index(
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
      Synapse_iterator<Input_synapse_interval> partial_input_iterator(solution.partial_solutions(partial_solution_iterator).input_data());
      const uint32 first_neuron_index_in_partial = solution.partial_solutions(partial_solution_iterator).output_data().starts();
      for( /* Skim through the inner neurons in the partial solutiomake n until the current one if found */
        uint32 i_neuron_iter = 0; i_neuron_iter < solution.partial_solutions(partial_solution_iterator).internal_neuron_number();++i_neuron_iter
      ){ /*!Note: i_neuron_iter == inner neuron iterator */
        if(neuron_iterator == static_cast<sint32>(first_neuron_index_in_partial + i_neuron_iter)){
          /* If the current neuron being checked is the one in the partial solution under i_neuron_iter */
          neuron_synapse_element_iterator = 0;

          /* Test iterates over the Neurons input weights, to see if they match with the wights in the Network */
          Synapse_iterator<> inner_neuron_weight_iterator(solution.partial_solutions(partial_solution_iterator).weight_indices());
          Synapse_iterator<> neuron_weight_iterator(net.neuron_array(neuron_iterator).input_weights());

          /* Inner Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator<>::is_index_input is true ) */
          expected_inputs = 0;
          counted_inputs = 0;
          inner_neuron_weight_iterator.iterate([&](Index_synapse_interval weight_synapse){
            expected_inputs += weight_synapse.interval_size();
          },[&](Index_synapse_interval weight_synapse, sint32 input_index){
            REQUIRE( neuron_weight_iterator.size() > neuron_synapse_element_iterator );
            CHECK(
              solution.partial_solutions(partial_solution_iterator).weight_table(input_index)
              == net.weight_table(neuron_weight_iterator[neuron_synapse_element_iterator])
            );
            ++counted_inputs;
            ++neuron_synapse_element_iterator;
          },weight_synapse_offset,solution.partial_solutions(partial_solution_iterator).weight_synapse_number(i_neuron_iter));

          /* Test if all of the neurons inputs are are the same as the ones in the net */
          /* Test iterates over the inner neurons synapse to see if it matches the Neuron synapse */
          Synapse_iterator<Input_synapse_interval> inner_neuron_input_iterator(solution.partial_solutions(partial_solution_iterator).inside_indices());
          Synapse_iterator<Input_synapse_interval> neuron_input_iterator(net.neuron_array(neuron_iterator).input_indices());

          /* Neuron inputs point to indexes in the partial solution input ( when Synapse_iterator<>::is_index_input s true ) */
          neuron_synapse_element_iterator = 0;
          counted_inputs = 0;
          inner_neuron_input_iterator.iterate([&](Input_synapse_interval input_synapse, sint32 input_index){
            REQUIRE( neuron_input_iterator.size() > neuron_synapse_element_iterator );
            if(!Synapse_iterator<>::is_index_input(input_index)){ /* Inner neuron takes its input internally */
              CHECK( 0 == input_synapse.reach_past_loops() ); /* Internal inputs should always be taken from the current loop */
              CHECK(
                static_cast<sint32>(first_neuron_index_in_partial + input_index)
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
            }else{ /* Inner Neuron takes its input from the partial solution input */
              CHECK( /* Input indices match */
                partial_input_iterator[Synapse_iterator<>::input_index_from_synapse_index(input_index)]
                == neuron_input_iterator[neuron_synapse_element_iterator]
              );
             CHECK( /* The time the neuron takes its input also match */
                partial_input_iterator.synapse_under(Synapse_iterator<>::input_index_from_synapse_index(input_index)).reach_past_loops()
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

void print_training_sample(uint32 sample_sequence_index, Data_aggregate& data_set, SparseNet& net, Service_context& service_context){
  Solution_solver sample_solver(*Solution_builder(service_context).build(net), service_context);
  vector<sdouble32> neuron_data(data_set.get_sequence_size());
  std::cout.precision(2);
  std::cout << std::endl << "Training sample["<< sample_sequence_index <<"]:" << std::endl;
  std::cout << std::endl << "..or raw_sample["<< (data_set.get_sequence_size() * sample_sequence_index) <<"]:" << std::endl;
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
  DataRingbuffer output_data(
    std::max(data_set.get_sequence_size(), sample_solver.get_solution().network_memory_length()),
    sample_solver.get_solution().neuron_number()
  );
  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "["<< data_set.get_label_sample((data_set.get_sequence_size() * sample_sequence_index) + j)[0] <<"]";
    sample_solver.solve(data_set.get_input_sample((data_set.get_sequence_size() * sample_sequence_index) + j), output_data);
    neuron_data[j] = output_data.get_element(0).back();
  }
  std::cout << std::endl;
  std::cout << "------<>------actual:" << std::endl;

  for(uint32 j = 0;j < data_set.get_sequence_size();++j){
    std::cout << "["<< neuron_data[j] <<"]";
  }
  std::cout << std::endl;
  std::cout << "==============" << std::endl;
  std::cout << "Neural memory for current sequence: " << std::endl;
  for(const vector<sdouble32>& vector : output_data.get_whole_buffer()){
    for(const sdouble32& element : vector) std::cout << "[" << element << "]";
    std::cout << std::endl;
  }

  std::cout << "weights: " << std::endl;
  for(int i = 0; i < net.weight_table_size(); ++i){
    std::cout << "[" << net.weight_table(i) << "]";
  }
  std::cout << std::endl;

  std::cout.precision(15);
}

Data_aggregate* create_addition_dataset(
  uint32 number_of_samples, SparseNet& net, cost_functions the_function, Service_context& service_context
){

  using std::vector;

  deque<vector<sdouble32>> net_inputs(number_of_samples);
  deque<vector<sdouble32>> addition_dataset(number_of_samples);

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
    addition_dataset[i].push_back(net_inputs[i][0] + net_inputs[i][1]);
  }

  return google::protobuf::Arena::Create<Data_aggregate>(
    service_context.get_arena_ptr(),
    service_context,
    deque<vector<sdouble32>>(net_inputs),
    deque<vector<sdouble32>>(addition_dataset),
    net, the_function
  );
}

Data_aggregate* create_sequenced_addition_dataset(
  uint32 number_of_samples, uint32 sequence_size, SparseNet& net, cost_functions the_function, Service_context& service_context
){
  uint32 carry_bit;
  deque<vector<sdouble32>> net_inputs(sequence_size * number_of_samples);
  deque<vector<sdouble32>> addition_dataset(sequence_size * number_of_samples);

  srand(time(nullptr));
  for(uint32 i = 0;i < number_of_samples;++i){
    carry_bit = 0;
    for(uint32 j = 0;j <sequence_size;++j){ /* Add testing and training sequences randomly */
      net_inputs[(sequence_size * i) + j] = vector<sdouble32>(2);
      addition_dataset[(sequence_size * i) + j] = vector<sdouble32>(1);
      net_inputs[(sequence_size * i) + j][0] = static_cast<sdouble32>(rand()%2);
      net_inputs[(sequence_size * i) + j][1] = static_cast<sdouble32>(rand()%2);

      addition_dataset[(sequence_size * i) + j][0] =
        net_inputs[(sequence_size * i) + j][0]
        + net_inputs[(sequence_size * i) + j][1]
        + carry_bit;
      if(1 < addition_dataset[(sequence_size * i) + j][0]){
        addition_dataset[(sequence_size * i) + j][0] = 1;
        carry_bit = 1;
      }else{
        carry_bit = 0;
      }
    }
  }

  return google::protobuf::Arena::Create<Data_aggregate>(
    service_context.get_arena_ptr(),
    service_context,
    deque<vector<sdouble32>>(net_inputs),
    deque<vector<sdouble32>>(addition_dataset),
    net, the_function, sequence_size
  );

}


} /* namsepace sparse_net_library_test */
