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

#include "rafko_utilities/services/thread_group.h"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/models/rafko_softmax_feature.h"

#include "test/test_utility.h"

namespace rafko_net_test {

static void check_softmax_values(std::vector<sdouble32>& neuron_data, rafko_net::FeatureGroup& mockup, rafko_utilities::ThreadGroup& threads){

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
  iter.iterate([&manual_sum, &neuron_data_copy](sint32 index){
    neuron_data_copy[index] /= manual_sum;
  }); /* normalizing e_x --> e_x/sum(e_x) */

  manual_sum = double_literal(0);
  iter.iterate([&manual_sum, &neuron_data_copy](sint32 index){
    manual_sum += neuron_data_copy[index];
  }); /* collecting softmax end result sum */
  REQUIRE( Catch::Approx(manual_sum).epsilon(double_literal(0.00000000000001)) == double_literal(1.0) );

  /* Calculate through the network */
  rafko_net::RafkoSoftmaxFeature::calculate(neuron_data, mockup.relevant_neurons(), threads);

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

TEST_CASE( "Checkig whether the softmax function is calculating correctly with whole arrays", "[features][softmax]" ){
  std::vector<sdouble32> neuron_data{double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0)};
  rafko_net::FeatureGroup mockup;
  rafko_utilities::ThreadGroup threads(4);
  rafko_net::IndexSynapseInterval tmp_interval;

  mockup.set_feature(rafko_net::neuron_group_feature_softmax);
  mockup.add_relevant_neurons()->set_interval_size(neuron_data.size());
  check_softmax_values(neuron_data, mockup, threads);

  neuron_data = {double_literal(1.0)};
  mockup.mutable_relevant_neurons(0)->set_interval_size(neuron_data.size());
  check_softmax_values(neuron_data, mockup, threads);

  for(uint32 variant = 0; variant < 10u; ++variant){
    neuron_data.clear(); /* set data to exaimne */
    for(sint32 i = 0; i < rand()%100; ++i){
      neuron_data.push_back( static_cast<sdouble32>(rand()%10)/10 );
    }

    mockup.mutable_relevant_neurons(0)->set_interval_size(neuron_data.size()); /* one synapse for the whole array */

    rafko_net::ThreadGroup loop_threads((rand()%16) + 1u);
    check_softmax_values(neuron_data, mockup, loop_threads);
  }
}

TEST_CASE( "Checkig whether the softmax function is calculating correctly with multiple random synapses inside an array", "[features][softmax]" ){
  std::vector<sdouble32> neuron_data{double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0),double_literal(0)};
  rafko_net::FeatureGroup mockup;
  rafko_utilities::ThreadGroup threads(4);
  rafko_net::IndexSynapseInterval tmp_interval;

  mockup.set_feature(rafko_net::neuron_group_feature_softmax);
  for(uint32 variant = 0; variant < 10u; ++variant){
    neuron_data.clear(); /* set data to examine */
    for(sint32 i = 0; i < rand()%200; ++i){
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

    rafko_net::ThreadGroup loop_threads((rand()%16) + 1u);
    check_softmax_values(neuron_data, mockup, loop_threads);
  }
}


} /* namespace rafko_net_test */
