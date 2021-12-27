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

#include "rafko_global.h"

#include <vector>
#include <stdexcept>

#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_utilities{

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
class RAFKO_FULL_EXPORT DataRingbuffer{
public:
  DataRingbuffer(uint32 buffer_number, uint32 buffer_size)
  :  current_index(0)
  ,  data(buffer_number)
  {
    for(std::vector<sdouble32>& buffer : data)
      buffer = std::vector<sdouble32>(buffer_size, double_literal(0.0));
  }

  /**
   * @brief      Store the current data and move the iterator forward for the next one
   */
  void step(){
    current_index = (current_index + 1)%(data.size());
    if(1 < data.size())
      std::copy(get_element(1).begin(),get_element(1).end(),get_element(0).begin());
  }

  /**
   * @brief      Resets every data element to all zeroes.
   */
  void reset(){
    current_index = (data.size()-1); /* Set the current index into the last index, so at the next @ */
    for(std::vector<sdouble32>& vector : data)
      for(sdouble32& element : vector) element = double_literal(0.0);
  }

  /**
   * @brief      Removes the first element from the Queue by
   *             filling the latest item with zeroes, and setting the
   *             current index one step back into the past.
   */
  void pop_front(){
    std::fill(get_element(0).begin(),get_element(0).end(),double_literal(0.0));
    current_index = get_buffer_index(1);
  }

  /**
   * @brief      Take over the latest row from the provided buffer
   *
   * @param[in]  other  The buffer to take the data from
   */
  void copy_latest(const DataRingbuffer& other){
    std::copy(
      other.get_const_element(0).begin(),other.get_const_element(0).end(),
      get_element(0).begin()
    );
  }

  /**
   * @brief      Gets the whole o the underlying data as a constant reference
   *
   * @return     The non-modifyable raw buffer data
   */
  const std::vector<std::vector<sdouble32>>& get_whole_buffer() const{
    return data;
  }

  /**
   * @brief      Gets adata value under the provided index parameters
   *
   * @param[in]  past_index  The past index
   * @param[in]  data_index  The index of the data point to retrive in the buffer
   *
   * @return     The value of the data in the   given index parameters
   */
  sdouble32 get_element(uint32 past_index, uint32 data_index) const{
    if((data.size() > past_index)&&(data[0].size() > data_index))
      return get_const_element(past_index)[data_index];
      else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

  /**
   * @brief      Sets the data element under the given indices to the provided value
   *
   * @param[in]  past_index  The buffer to set the data
   * @param[in]  data_index  The index of the data inside the buffer
   * @param[in]  value       The value to overwrite the data with
   */
  void set_element(uint32 past_index, uint32 data_index, sdouble32 value){
    if((data.size() > past_index)&&(data[0].size() > data_index))
      get_element(past_index)[data_index] = value;
      else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

  /**
   * @brief      Gets a reference to a stored entry in the ringbuffer
   *
   * @param[in]  past_index  The past index
   *
   * @return     The reference pointing to a data
   */
  std::vector<sdouble32>& get_element(uint32 past_index){
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
  const std::vector<sdouble32>& get_const_element(uint32 past_index) const{
    if(past_index < data.size()){
      return data[get_buffer_index(past_index)];
    }else throw std::runtime_error("Ringbuffer index out of bounds!");
  }

  /**
   * @brief      Gets the number of buffers stored in the object
   *
   * @return     The sequence size.
   */
  uint32 get_sequence_size() const{
    return data.size();
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
   * @brief      Returns the size of the available memory buffers
   *
   * @return     Number of elements
   */
  uint32 buffer_size() const{
    return data[0].size();
  }

  /**
   * @brief      Returns the number of available memory buffer
   *
   * @return     number of buffers available
   */
  uint32 buffer_number() const{
    return data.size();
  }

private:
  uint32 current_index;
  std::vector<std::vector<sdouble32>> data;

  /**
   * @brief      Gets the buffer index for the given past index
   *
   * @param[in]  past_index  The past index
   *
   * @return     The buffer index.
   */
  uint32 get_buffer_index(uint32 past_index) const{
    if(data.size() <= past_index) throw std::runtime_error("Older data queried, than memory capacity.");
    if(past_index > current_index) return (data.size() + current_index - past_index);
      else return(current_index - past_index);
  }
};

} /* namespace rafko_utilities */

#endif /* DATA_RINGBUFFER_H */
