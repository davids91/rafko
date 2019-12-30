#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"
#include "services/synapse_iterator.h"



namespace sparse_net_library_test {

using std::abs;
using std::vector;

using sparse_net_library::uint32;
using sparse_net_library::sint32;
using sparse_net_library::Neuron;
using sparse_net_library::Synapse_iterator;
using sparse_net_library::Synapse_interval;


/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
 *  indexes
 */
TEST_CASE("Synapse Iteration","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
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
  Synapse_iterator iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](int index){
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
TEST_CASE("Synapse iteration on a range","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
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
  Synapse_iterator iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](int index){
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
TEST_CASE("Synapse iteration including negative numbers","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
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
  Synapse_iterator iter(neuron.input_indices());

  REQUIRE( 110 == iter.size() );

  iter.iterate([&](int index){
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
TEST_CASE("Synapse Iterator direct access","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  Synapse_iterator iter(neuron.input_indices());
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
TEST_CASE("Synapse Iterator Skimming","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  Synapse_iterator iter(neuron.input_indices());

  sint32 manual_index = 0;
  iter.skim([&](int synapse_start, unsigned int synapse_size){
    CHECK( synapse_start == synapse_indexes[manual_index][0] );
    CHECK( synapse_size == synapse_indexes[manual_index][1] );
    ++manual_index;
  });
}

/*###############################################################################################
 * Testing synapse utility functions
 * - Creating an artificial synapse pair, testing .size and .back
 */
TEST_CASE("Synapse Iterator Utility functions","[synapse_iteration]"){
  Neuron neuron = Neuron();
  Synapse_interval temp_synapse_interval;
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{70,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  Synapse_iterator iter(neuron.input_indices());
  CHECK( 110 == iter.size() );
  CHECK( -89 == iter.back() );
}


} /* namespace sparse_library_test */
