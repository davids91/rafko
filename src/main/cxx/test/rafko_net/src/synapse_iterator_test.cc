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

#include "test/test_utility.h"

#include <catch2/catch_test_macros.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/services/synapse_iterator.h"

namespace rafko_net_test {

using std::abs;
using std::vector;

using rafko_net::Neuron;
using rafko_net::SynapseIterator;
using rafko_net::InputSynapseInterval;
using rafko_net::IndexSynapseInterval;


/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
 *  indexes
 */
TEST_CASE("Synapse Iteration","[synapse-iteration]"){
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());

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
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 1;
  sint32 manual_index = synapse_indexes[1][0];
  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());

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
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{-60,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](sint32 index){
    REQUIRE( synapse_indexes.size() > range_iter );
    CHECK( index == manual_index );
    --manual_index;
    if(
      ((synapse_indexes.size()-1) > range_iter) /* Only adapt range iter, if there is a next range element */
      &&(abs(manual_index - synapse_indexes[range_iter][0]) >= synapse_indexes[range_iter][1])
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
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());
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
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());

  sint32 manual_index = 0;
  iter.skim([&](InputSynapseInterval input_synapse){
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
  Neuron neuron = Neuron();
  InputSynapseInterval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  SynapseIterator<InputSynapseInterval> iter(neuron.input_indices());
  CHECK( 110 == iter.size() );
  CHECK( -89 == iter.back() );
}

/*###############################################################################################
 * Testing if synapse iteration is valid with given ranges as well
 * - Create an artificial synapse array of at least 3 elements
 * - See if the iteration goes correctly for each
 */
TEST_CASE("Ranged Synapse iteration","[synapse-iteration]"){
  Neuron neuron = Neuron();
  IndexSynapseInterval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {50,3},{70,3},{20,2},{30,2}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_weights() = temp_synapse_interval;
  }

  SynapseIterator<> iter(neuron.input_weights());
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


} /* namespace sparse_library_test */
