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

#include <vector>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/services/thread_group.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_mainframe/models/rafko_settings.h"
#include "rafko_net/services/rafko_network_feature.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_net/services/solution_solver.h"

#include "test/test_utility.h"

namespace rafko_net_test {

namespace{
  void check_softmax_values(
    std::vector<sdouble32>& neuron_data, const rafko_mainframe::RafkoSettings& settings,
    rafko_net::FeatureGroup& mockup, rafko_utilities::ThreadGroup& threads
  ){

    /* calculate manually */
    rafko_net::SynapseIterator<> iter(mockup.relevant_neurons());
    std::vector<sdouble32> neuron_data_copy = std::vector(neuron_data);
    sdouble32 max_value = -std::numeric_limits<double>::max();
    iter.iterate([&max_value, &neuron_data_copy](sint32 index){
      if(neuron_data_copy[index] > max_value)
        max_value = neuron_data_copy[index];
    }); /* finding maximum value */
    iter.iterate([&max_value, &neuron_data_copy](sint32 index){
      neuron_data_copy[index] = std::exp(neuron_data_copy[index] - max_value);
    }); /* transforming x --> exp(x - max(x)) */

    sdouble32 manual_sum = double_literal(0);
    iter.iterate([&manual_sum, &neuron_data_copy](sint32 index){
      manual_sum += neuron_data_copy[index];
    }); /* collecting resulting sum */
    manual_sum = std::max(manual_sum, std::numeric_limits<double>::epsilon());
    iter.iterate([&manual_sum, &neuron_data_copy](sint32 index){
      neuron_data_copy[index] /= manual_sum;
    }); /* normalizing e_x --> e_x/sum(e_x) */

    manual_sum = double_literal(0);
    iter.iterate([&manual_sum, &neuron_data_copy](sint32 index){
      manual_sum += neuron_data_copy[index];
    }); /* collecting softmax end result sum */
    REQUIRE( Catch::Approx(manual_sum).epsilon(double_literal(0.00000000000001)) == double_literal(1.0) );

    /* Calculate through the network */
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> exec_threads;
    exec_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(threads.get_number_of_threads()));
    rafko_net::RafkoNetworkFeature features(exec_threads);
    features.execute_solution_relevant(mockup, settings, neuron_data);

    /* Check if sum equals to 1 */
    sdouble32 sum = double_literal(0);
    iter.iterate([&sum, &neuron_data_copy](sint32 index){
      sum += neuron_data_copy[index];
    }); /* collecting softmax end result sum */
    CHECK( Catch::Approx(sum).epsilon(double_literal(0.00000000000001)) == double_literal(1.0) );

    /* check if each element equal ( exp(x) / sum(exp(x)) ) */
    for(uint32 i = 0; i < neuron_data.size(); i++){
      REQUIRE( Catch::Approx(neuron_data[i]).epsilon(double_literal(0.00000000000001)) == neuron_data_copy[i] );
    }

  }
} /* namespace */

TEST_CASE( "Checkig whether the softmax function is calculating correctly with whole arrays", "[features][softmax]" ){
  rafko_mainframe::RafkoSettings settings;
  std::vector<sdouble32> neuron_data{double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0)};
  rafko_net::FeatureGroup mockup;
  rafko_utilities::ThreadGroup threads(4);
  rafko_net::IndexSynapseInterval tmp_interval;

  mockup.set_feature(rafko_net::neuron_group_feature_softmax);
  mockup.add_relevant_neurons()->set_interval_size(neuron_data.size());
  check_softmax_values(neuron_data, settings, mockup, threads);

  neuron_data = {double_literal(1.0)};
  mockup.mutable_relevant_neurons(0)->set_interval_size(neuron_data.size());
  check_softmax_values(neuron_data, settings, mockup, threads);

  for(uint32 variant = 0; variant < 10u; ++variant){
    neuron_data.clear(); /* set data to exaimne */
    for(sint32 i = 0; (i < rand()%100)||(0 == neuron_data.size()); ++i){
      neuron_data.push_back( static_cast<sdouble32>(rand()%10)/10 );
    }

    mockup.mutable_relevant_neurons(0)->set_interval_size(neuron_data.size()); /* one synapse for the whole array */

    rafko_utilities::ThreadGroup loop_threads((rand()%16) + 1u);
    check_softmax_values(neuron_data, settings, mockup, loop_threads);
  }
}

TEST_CASE( "Checkig whether the softmax function is calculating correctly with multiple random synapses inside an array", "[features][softmax]" ){
  rafko_mainframe::RafkoSettings settings;
  std::vector<sdouble32> neuron_data{double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0)};
  rafko_net::FeatureGroup mockup;
  rafko_utilities::ThreadGroup threads(4);
  rafko_net::IndexSynapseInterval tmp_interval;
  mockup.set_feature(rafko_net::neuron_group_feature_softmax);
  for(uint32 variant = 0; variant < 10u; ++variant){
    neuron_data.clear(); /* set data to examine */
    for(sint32 i = 0; ((i < rand()%200)||(0u == neuron_data.size())); ++i){
      neuron_data.push_back( static_cast<sdouble32>(rand()%10)/10 );
    }

    mockup.mutable_relevant_neurons()->Clear(); /* adding random synapses */
    uint16 number_of_synapses = std::max(static_cast<uint16>(1u), static_cast<uint16>(rand()%(neuron_data.size())));
    uint16 start_index = 0; /* starting from the beginning of the array */
    for(uint16 synapse_iterator = 0; synapse_iterator < number_of_synapses; ++synapse_iterator){
      uint16 synapse_size = rand()%(neuron_data.size() - start_index)/2; /* decide the size of the current synapse */
      synapse_size = std::max(static_cast<uint16>(1u), synapse_size);
      if((start_index + synapse_size) > neuron_data.size())break; /* do not reach beyond the available array */
      tmp_interval.set_starts(start_index);
      tmp_interval.set_interval_size(synapse_size);
      *mockup.add_relevant_neurons() = tmp_interval; /* add the synapse */
      start_index += synapse_size; /* nex synapse can start after the current one */
      if(start_index >= neuron_data.size())break;
      start_index += rand()%(neuron_data.size() - start_index)/2; /* add a gap to the synapses randomly for roboustness */
      if((start_index + synapse_size) > neuron_data.size())break; /* do not reach beyond the available array */
    }

    rafko_utilities::ThreadGroup loop_threads((rand()%16) + 1u);
    check_softmax_values(neuron_data, settings, mockup, loop_threads);
  }
}

TEST_CASE("Checking if the network builder is correctly placing the softmax feature into the built network", "[features][build][manual]"){
  rafko_mainframe::RafkoSettings settings;
  std::unique_ptr<rafko_net::RafkoNet> net = std::unique_ptr<rafko_net::RafkoNet>(
    rafko_net::RafkoNetBuilder(settings).input_size(5)
      .expected_input_range(double_literal(5.0))
      .add_feature_to_layer(0, rafko_net::neuron_group_feature_softmax)
      .add_feature_to_layer(2, rafko_net::neuron_group_feature_softmax)
      .dense_layers({20,40,30,10,20})
  );

  CHECK( 2 == net->neuron_group_features_size() );

  CHECK( 1 == net->neuron_group_features(0).relevant_neurons_size() );
  CHECK( 0 == net->neuron_group_features(0).relevant_neurons(0).starts() );
  CHECK( 20 == net->neuron_group_features(0).relevant_neurons(0).interval_size() );

  CHECK( 1 == net->neuron_group_features(0).relevant_neurons_size() );
  CHECK( 60 == net->neuron_group_features(1).relevant_neurons(0).starts() );
  CHECK( 30 == net->neuron_group_features(1).relevant_neurons(0).interval_size() );

  net.reset();
}

TEST_CASE("Checking if the network builder is correctly placing the softmax feature into the built network", "[features][build]"){
  rafko_mainframe::RafkoSettings settings;
  std::unique_ptr<rafko_net::RafkoNet> net;
  for(uint32 variant = 0; variant < 10u; ++variant){
    std::vector<uint32> net_structure;
    while((rand()%10 < 9)||(4 > net_structure.size()))
      net_structure.push_back(static_cast<uint32>(rand()%30) + 1u);

    uint8 num_of_features = rand()%(net_structure.size()/2) + 1u;
    rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
      .input_size(5)
      .expected_input_range(double_literal(5.0));

    uint8 layer_of_feature_index = 0;
    std::vector<uint32> feature_neuron_start_index;
    std::vector<uint32> feature_layer;
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
      feature_neuron_start_index.push_back(layer_start_index);
      feature_layer.push_back(layer_of_feature_index);
    }
    net = std::unique_ptr<rafko_net::RafkoNet>(builder.dense_layers(net_structure));
    for(const rafko_net::FeatureGroup& feature : net->neuron_group_features()){ /* check if all the features point to the correct neuron indices */
      REQUIRE( 1u == feature.relevant_neurons_size() );
      REQUIRE( feature.relevant_neurons(0u).starts() == static_cast<sint32>(feature_neuron_start_index.front()) );
      REQUIRE( feature.relevant_neurons(0u).interval_size() == net_structure[feature_layer.front()] );
      feature_neuron_start_index.erase(feature_neuron_start_index.begin());
      feature_layer.erase(feature_layer.begin());
    }
    net.reset();
  }
}

TEST_CASE("Checking if the Solution Builder is correctly producing the softmax features in the solution builder", "[features][build]"){
  rafko_mainframe::RafkoSettings settings;
  std::unique_ptr<rafko_net::RafkoNet> net;
  std::unique_ptr<rafko_net::Solution> solution;

  for(uint32 variant = 0; variant < 10u; ++variant){
    net = std::unique_ptr<rafko_net::RafkoNet>(rafko_test::generate_random_net_with_softmax_features(3, settings));
    solution = std::unique_ptr<rafko_net::Solution>(rafko_net::SolutionBuilder(settings).build(*net));

    /* every softmax feature inside the @RafkoNet should be found inside the @Solution */
    for(const rafko_net::FeatureGroup& feature : net->neuron_group_features()){
      bool found = false;
      for(const rafko_net::PartialSolution& partial : solution->partial_solutions()){
        for(const rafko_net::FeatureGroup& partial_feature : partial.solved_features()){
          if(
            (feature.feature() == partial_feature.feature())
            &&(rafko_net::SynapseIterator<>(feature.relevant_neurons()) == rafko_net::SynapseIterator<>(partial_feature.relevant_neurons()))
          ){
            found = true;
            goto Found_feature_in_partials;
          }
        }/* for( every solved feature in partial solution ) */
      }/* for( every partial in solution ) */
      Found_feature_in_partials:;
      REQUIRE( found == true );
    }/* for( every feature in network ) */
    net.reset();
  }
}

TEST_CASE("Checking if the network solver is correctly producing the softmax feature values through the solution solver", "[features][build]"){
  rafko_mainframe::RafkoSettings settings;
  std::unique_ptr<rafko_net::RafkoNet> net;
  std::unique_ptr<rafko_net::Solution> solution;
  std::unique_ptr<rafko_net::SolutionSolver> solver;
  for(uint32 variant = 0; variant < 10u; ++variant){
    net = std::unique_ptr<rafko_net::RafkoNet>(rafko_test::generate_random_net_with_softmax_features(3, settings));
    solution = std::unique_ptr<rafko_net::Solution>(rafko_net::SolutionBuilder(settings).build(*net));
    solver = std::unique_ptr<rafko_net::SolutionSolver>(rafko_net::SolutionSolver::Builder(*solution, settings).build());

    (void)solver->solve({double_literal(0),double_literal(6),double_literal(5)});
    rafko_utilities::DataRingbuffer neuron_data = solver->get_memory();
    for(const rafko_net::FeatureGroup& feature : net->neuron_group_features()){ /* check if all the features point to the correct neuron indices */
      sdouble32 sum = double_literal(0.0);
      rafko_net::SynapseIterator<>::iterate(feature.relevant_neurons(),[&sum, neuron_data](sint32 neuron_index){
        sum += neuron_data.get_element(0u, neuron_index);
      });
      REQUIRE( Catch::Approx(sum).epsilon(double_literal(0.00000000000001)) == double_literal(1.0) );
    }
    net.reset();
  }
}

} /* namespace rafko_net_test */
