#include "test/catch.hpp"

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"
#include "services/synapse_iterator.h"



namespace sparse_net_library_test {

using std::abs;
using std::vector;

using sparse_net_library::uint32;
using sparse_net_library::sint32;
using sparse_net_library::Neuron;
using sparse_net_library::Synapse_iterator;


  /*###############################################################################################
   * Testing synapse iteration
   * - Creating an artificial synapse pair, and testing if the indexes follow the laid out
   *  indexes
   */

TEST_CASE("Synapse Iteration","[neuron_iteration]"){
  Neuron neuron = Neuron();
  vector<vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    neuron.add_input_index_starts(synapse_indexes[i][0]);
    neuron.add_input_index_sizes(synapse_indexes[i][1]);
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  Synapse_iterator iter(neuron.input_index_starts(),neuron.input_index_sizes());
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
    return true;
  });
}

TEST_CASE("Synapse iteration on a range","[neuron_iteration]"){
  Neuron neuron = Neuron();
  vector<vector<uint32>> synapse_indexes = {
    {50,10},{60,30},{20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    neuron.add_input_index_starts(synapse_indexes[i][0]);
    neuron.add_input_index_sizes(synapse_indexes[i][1]);
  }

  uint32 range_iter = 1;
  sint32 manual_index = synapse_indexes[1][0];
  Synapse_iterator iter(neuron.input_index_starts(),neuron.input_index_sizes());
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
    return true;
  },1,2);
}

TEST_CASE("Synapse iteration including negative numbers","[neuron_iteration]"){
  Neuron neuron = Neuron();
  vector<vector<sint32>> synapse_indexes = {
    {-50,10},{-60,30},{-20,70}
  }; /* {{range},{start,length},{range}..} */

  for(uint32 i = 0; i < synapse_indexes.size(); ++i){
    neuron.add_input_index_starts(synapse_indexes[i][0]);
    neuron.add_input_index_sizes(synapse_indexes[i][1]);
  }

  uint32 range_iter = 0;
  sint32 manual_index = synapse_indexes[0][0];
  Synapse_iterator iter(neuron.input_index_starts(),neuron.input_index_sizes());
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
    return true;
  });
}

} /* namespace sparse_library_test */
