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

#include <vector>

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
  :  loop_index(0)
  ,  current_index(loop_index)
  ,  data(buffer_number)
  {
    for(vector<sdouble32>& buffer : data)
      buffer = vector<sdouble32>(buffer_size, double_literal(0.0));
  }

  /**
   * @brief      Store the current data and move the iterator forward for the next one
   */
  void step(){
    current_index = (current_index + 1)%(data.size());
    if(1 < data.size())
      std::copy(get_element(1).begin(),get_element(1).end(),get_element(0).begin());
  }

  sdouble32 get_element(uint32 data_index, uint32 past_index){
    if(data[0].size() > data_index)return get_element(past_index)[data_index];
     else throw "Ringbuffer data index out of bounds!";
  }

  /**
   * @brief      Gets a reference to a stored entry in the ringbuffer
   *
   * @param[in]  past_index  The past index
   *
   * @return     The reference pointing to a data
   */
  vector<sdouble32>& get_element(uint32 past_index){
    loop_index = current_index;
    if(past_index < data.size()){
      for(uint32 i = 0; i < past_index; ++i)
        if(0 < loop_index)--loop_index;
          else loop_index = data.size()-1;
      return data[loop_index];
    }else throw "Ringbuffer index out of bounds!";
  }

private:
  uint32 loop_index;
  uint32 current_index;
  vector<vector<sdouble32>> data;
};

} /* namespace sparse_net_library */

#endif /* DATA_RING_BUFFER_H */
