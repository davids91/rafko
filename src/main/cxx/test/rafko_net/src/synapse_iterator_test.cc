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

#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/synapse_iterator.h"

#include "test/test_utility.h"

namespace rafko_net_test {

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
 *  indexes
 */
TEST_CASE("Synapse Iteration","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](sint32 index){
    REQUIRE( synapse_indexes.size() > range_iter );
    CHECK( index == manual_index );
    ++manual_index;
    if(
      ((synapse_indexes.size()-1) > range_iter) /* Only adapt range iter, if there is a next range element */
      &&(manual_index - synapse_indexes[range_iter][0]) >= synapse_indexes[range_iter][1]
    ){
      ++range_iter;
      manual_index = synapse_indexes[range_iter][0];
    }
  });
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
 *  indexes, based on ranges
 */
TEST_CASE("Synapse iteration on a range","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 1;
  sint32 manual_index = synapse_indexes[1][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](sint32 index){
    REQUIRE( synapse_indexes.size() > range_iter );
    CHECK( index == manual_index );
    ++manual_index;
    if(
      ((synapse_indexes.size()-1) > range_iter) /* Only adapt range iter, if there is a next range element */
      &&(manual_index - synapse_indexes[range_iter][0]) >= synapse_indexes[range_iter][1]
    ){
      ++range_iter;
      manual_index = synapse_indexes[range_iter][0];
    }
  },1,2);
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
 *  indexes, even with negative numbers
 */
TEST_CASE("Synapse iteration including negative numbers","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<sint32>> synapse_indexes = {
    {-50,10},{-60,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](sint32 index){
    REQUIRE( synapse_indexes.size() > range_iter );
    CHECK( index == manual_index );
    --manual_index;
    if(
      ((synapse_indexes.size()-1) > range_iter) /* Only adapt range iter, if there is a next range element */
      &&(std::abs(manual_index - synapse_indexes[range_iter][0]) >= synapse_indexes[range_iter][1])
    ){
      ++range_iter;
      manual_index = synapse_indexes[range_iter][0];
    }
  });
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if operator[] is reaching the correct indexes
 *   and correctly mapping the synapse inputs into a contigous array
 */
TEST_CASE("Synapse Iterator direct access","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());
  CHECK( iter[0] == -50 );
  CHECK( iter[5] == -55 );
  CHECK( iter[10] == 70 );
  CHECK( iter[11] == 71 );
  CHECK( iter[12] == 72 );
  CHECK( iter[39] == 99 );
  CHECK( iter[40] == -20 );
  CHECK( iter[109] == -89 );
}


/*###############################################################################################
 * Testing synapse Skimming
 * - Creating an artificial synapse pair, and testing if skim operation goes through all the
 *   synapses, correctly displaying starting indices and sizes
 */
TEST_CASE("Synapse Iterator Skimming","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());

  sint32 manual_index = 0;
  iter.skim([&](rafko_net::InputSynapseInterval input_synapse){
    CHECK( input_synapse.starts() == synapse_indexes[manual_index][0] );
    CHECK( static_cast<sint32>(input_synapse.interval_size()) == synapse_indexes[manual_index][1] );
    ++manual_index;
  });
}

/*###############################################################################################
 * Testing synapse utility functions
 * - Creating an artificial synapse pair, testing .size and .back
 */
TEST_CASE("Synapse Iterator Utility functions","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(neuron.input_indices());
  CHECK( 110 == iter.size() );
  CHECK( -89 == iter.back() );
}

/*###############################################################################################
 * Testing if synapse iteration is valid with given ranges as well
 * - Create an artificial synapse array of at least 3 elements
 * - See if the iteration goes correctly for each
 */
TEST_CASE("Ranged Synapse iteration","[synapse-iteration]"){
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::IndexSynapseInterval temp_synapse_interval;
  std::vector<std::vector<sint32>> synapse_indexes = {
    {50,3},{70,3},{20,2},{30,2}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_weights() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<> iter(neuron.input_weights());
  uint32 iterate_range_number;
  sint32 iteration_count;
  uint32 current_synapse;
  sint32 element_index;

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){ /* go through each synapse interval */
    element_index = synapse_indexes[i][0]; /* the start of the synapse currently being iterated over */
    iteration_count = 0;
    iter.iterate([&](sint32 synapse_index){
      CHECK( element_index == synapse_index );
      ++element_index;
      ++iteration_count;
    },i,1);

    CHECK(iteration_count == synapse_indexes[i][1]);
  }

  sint32 iteration_in_this_synapse;
  for(uint32 i = 0; i < synapse_indexes.size(); ++i){ /* go through each synapse interval */
    current_synapse = i;
    iterate_range_number = std::min(2u,static_cast<uint32>(synapse_indexes.size()-i));
    element_index = synapse_indexes[i][0]; /* the start of the synapse currently being iterated over */
    iteration_count = 0;
    iteration_in_this_synapse = 0;
    iter.iterate([&](sint32 synapse_index){
      if(iteration_in_this_synapse >= synapse_indexes[current_synapse][1]){
        ++current_synapse;
        iteration_in_this_synapse = 0;
        element_index = synapse_indexes[current_synapse][0];
      }

      CHECK( element_index == synapse_index );

      ++iteration_in_this_synapse;
      ++element_index;
      ++iteration_count;
    },i,iterate_range_number);

    for(uint32 j = 0; j < iterate_range_number; ++j){
      iteration_count -= synapse_indexes[i+j][1];
    }

    CHECK( 0 == iteration_count );
  }
}

/*###############################################################################################
 * Testing if synapse iterator equality operator produces correct output
 */
TEST_CASE("Synapseiterator equaly","[synapse-iteration]"){
  rafko_net::Neuron neuron1 = rafko_net::Neuron();
  rafko_net::Neuron neuron2 = rafko_net::Neuron();
  rafko_net::Neuron neuron3 = rafko_net::Neuron();
  uint32 num_synapses = (rand()%50) + 1;
  for(uint32 synapse_index = 0; synapse_index < num_synapses; ++synapse_index){
    rafko_net::IndexSynapseInterval temp_interval;
    temp_interval.set_starts(rand()%435);
    temp_interval.set_interval_size(rand()%435);
    *neuron1.add_input_weights() = temp_interval;
    *neuron2.add_input_weights() = temp_interval;
    temp_interval.set_starts(500u + rand()%435);
    temp_interval.set_interval_size(200u + rand()%435);
    *neuron3.add_input_weights() = temp_interval;
  }

  CHECK( rafko_net::SynapseIterator<>(neuron1.input_weights()) == rafko_net::SynapseIterator<>(neuron2.input_weights()) );
  CHECK( rafko_net::SynapseIterator<>(neuron1.input_weights()) != rafko_net::SynapseIterator<>(neuron3.input_weights()) );
  CHECK( rafko_net::SynapseIterator<>(neuron2.input_weights()) != rafko_net::SynapseIterator<>(neuron3.input_weights()) );

  num_synapses = (rand()%50) + 1;
  for(uint32 synapse_index = 0; synapse_index < num_synapses; ++synapse_index){
    rafko_net::IndexSynapseInterval temp_interval;
    temp_interval.set_starts(rand()%435);
    temp_interval.set_interval_size(rand()%435);
    *neuron1.add_input_weights() = temp_interval;
    temp_interval.set_starts(500u + rand()%435);
    temp_interval.set_interval_size(200u + rand()%435);
    *neuron2.add_input_weights() = temp_interval;
    *neuron3.add_input_weights() = temp_interval;
  }

  CHECK( rafko_net::SynapseIterator<>(neuron1.input_weights()) != rafko_net::SynapseIterator<>(neuron2.input_weights()) );
  CHECK( rafko_net::SynapseIterator<>(neuron1.input_weights()) != rafko_net::SynapseIterator<>(neuron3.input_weights()) );
  CHECK( rafko_net::SynapseIterator<>(neuron2.input_weights()) != rafko_net::SynapseIterator<>(neuron3.input_weights()) );
}

} /* namespace sparse_library_test */
