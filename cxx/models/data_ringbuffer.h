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

#ifndef DATA_RINGBUFFER_H
#define DATA_RINGBUFFER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"

#include <vector>
#include <stdexcept>

namespace sparse_net_library{

using std::vector;

/**
 * @brief      This class describes a ringbuffer designed to store the Memory of a Neural Network.
 *             At the life-cycle of a Neural network one solution counts as a "loop", where the data 
 *             of the neurons shall be calculated and copied into an array. The array stores the activation values
 *             of the Neurons from this loop, and the previous loops as well. 
 *             This class provides that array. 
 *             At every loop, it provides Read/Write Access to the latest element in the buffer and
 *             read access to the previous data of the previous. At the start of each loop, it copies 
 *             the data from the previous loops into the current one. The data under the current loop 
 *             needs to keep its contents, while the data from previous loops also need to be stored. 
 */
class Data_ringbuffer{
public:
  Data_ringbuffer(uint32 buffer_number, uint32 buffer_size)
  :  current_index(0)
  ,  data(buffer_number)
  {
    for(vector<sdouble32>& buffer : data)
      buffer = vector<sdouble32>(buffer_size, double_literal(0.0));
  }

  /**
   * @brief      Store the current data and move the iterator forward for the next one
   */
  void step(void){
    current_index = (current_index + 1)%(data.size());
    if(1 < data.size())
      std::copy(get_element(1).begin(),get_element(1).end(),get_element(0).begin());
  }

  /**
   * @brief      Resets every data element to all zeroes.
   */
  void reset(void){
    for(vector<sdouble32>& vector : data)
      for(sdouble32& element : vector) element = double_literal(0.0);
  }
  
  /**
   * @brief      Removes the first element from the Queue by 
   *             filling the latest item with zeroes, and setting the 
   *             current index one step back into the past.
   */
  void pop_front(void){
    std::fill(get_element(0).begin(),get_element(0).end(),double_literal(0.0));
    current_index = get_buffer_index(1);
  }

  /**
   * @brief      Take over the latest row from the provided buffer
   *
   * @param[in]  other  The buffer to take the data from
   */
  void copy_latest(const Data_ringbuffer& other){
    std::copy(
      other.get_const_element(0).begin(),other.get_const_element(0).end(),
      get_element(0).begin()
    );
  }

  /**
   * @brief      Gets the data element in the..
   *
   * @param[in]  data_index  ..past @past_index th loop .. 
   * @param[in]  past_index  .. under index @data_index
   *
   * @return     The element.
   */
  sdouble32 get_element(uint32 data_index, uint32 past_index) const{
    if(data[0].size() > data_index)return get_const_element(past_index)[data_index];
     else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

  /**
   * @brief      Gets a reference to a stored entry in the ringbuffer
   *
   * @param[in]  past_index  The past index
   *
   * @return     The reference pointing to a data
   */
  vector<sdouble32>& get_element(uint32 past_index){
    if(past_index < data.size()){
      return data[get_buffer_index(past_index)];
    }else throw std::runtime_error("Ringbuffer index out of bounds!");
  }

  /**
   * @brief      Gets a const reference to a stored entry in the ringbuffer
   *
   * @param[in]  past_index  The past index
   *
   * @return     The reference pointing to a data
   */
  const vector<sdouble32>& get_const_element(uint32 past_index) const{
    if(past_index < data.size()){
      return data[get_buffer_index(past_index)];
    }else throw std::runtime_error("Ringbuffer index out of bounds!");
  }

  /**
   * @brief      Utility function to get an element from the given sequence. 
   *             Please also refer to the description of @get_sequence_index
   *
   * @param[in]  sequence_index             The sequence index
   * @param[in]  input_synapse              The input synapse
   * @param[in]  element_offset_from_start  The element offset from start
   *
   * @return     The buffer element in the given sequence
   */
  sdouble32 get_const_element(uint32 sequence_index, Input_synapse_interval input_synapse, uint32 element_offset_from_start) const{
    if(static_cast<sint32>(get_sequence_size()) > get_sequence_index(sequence_index,input_synapse)){
      if(input_synapse.starts() + element_offset_from_start < data[get_sequence_index(sequence_index,input_synapse)].size())
        return get_const_element(
          get_sequence_index(sequence_index,input_synapse)
        )[input_synapse.starts() + element_offset_from_start];
      else throw std::runtime_error("Buffer element index out of bounds!");
    }else return 0.0;
  }

  /**
   * @brief      Utility element to get a buffer under the given sequence based on a past reach back value
   *             provided in the given input synapse
   *
   * @param[in]  sequence_index  The sequence index
   * @param[in]  input_synapse   The input synapse
   *
   * @return     The constant element.
   */
  const vector<sdouble32>& get_const_element(uint32 sequence_index, Input_synapse_interval input_synapse) const{
    if(static_cast<sint32>(get_sequence_size()) > get_sequence_index(sequence_index,input_synapse)){
        return get_const_element(get_sequence_index(sequence_index,input_synapse));
    }else throw std::runtime_error("Buffer index out of bounds!");
  }

  /**
   * @brief      Gets the number of buffers stored in the object
   *
   * @return     The sequence size.
   */
  uint32 get_sequence_size(void) const{
    return data.size();
  }

  /**
   * @brief      Calculates the index to reach the neuron data at the @sequence_index th
   *             evaluation of a data sample which was the last @past_index th loop.
   *             Since the evaluation goes from the 0th item in the sequence,
   *             by the time every sequence is evaluated, the neuron_data_sequences buffer
   *             should contain the hidden data form every Neuron at the end of every solution.
   *             At that point, the "latest" neuron data array should be under
   *             neuron_data_sequences.get_element(0).
   *             If a Network is recurrent (has inputs "from the past"), what it would have
   *             as input at the last sequence is at index 0. If it would have input from 
   *             "the past", the inputs from the past would be under
   *             neuron_data_sequences.get_element(past_index). 
   *             In case the following data needs to be accessed: what would the network see as input 
   *             data in the @sequence_index 'th step, if that network would take input from the 
   *             past ( @past_index loops before that step ) => the below function shall be used.
   *
   * @param[in]  sequence_index  The sequence index
   * @param[in]  input_synapse   The input synapse containing the used past index
   *
   * @return     The buffer index.
   */
  sint32 get_sequence_index(uint32 sequence_index, Input_synapse_interval input_synapse) const{
    return get_sequence_index(sequence_index, input_synapse.reach_past_loops());
  }

  /**
   * @brief      Calculates the index to reach the neuron data at the @sequence_index th
   *             evaluation of a data sample which was the last @past_index th loop.
   *
   * @param[in]  sequence_index    The sequence index
   * @param[in]  reach_past_loops  The number of loops to reach back in the sequence
   *
   * @return     The buffer index.
   */
  sint32 get_sequence_index(uint32 sequence_index, uint32 reach_past_loops) const{
    return (get_sequence_size() - sequence_index - 1) + reach_past_loops;
  }

  /**
   * @brief      Returns the size of the available vectors
   *
   * @return     Number of elements
   */
  uint32 buffer_size(void) const{
    return data[0].size();
  }

private:
  uint32 current_index;
  vector<vector<sdouble32>> data;

  /**
   * @brief      Gets the buffer index for the given past index
   *
   * @param[in]  past_index  The past index
   *
   * @return     The buffer index.
   */
  uint32 get_buffer_index(uint32 past_index) const{
    uint32 loop_index = current_index;
    for(uint32 i = 0; i < past_index; ++i)
      if(0 < loop_index)--loop_index;
        else loop_index = data.size()-1;
    return loop_index;
  }
};

} /* namespace sparse_net_library */

#endif /* DATA_RING_BUFFER_H */
