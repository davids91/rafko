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

#include "rafko_global.hpp"

#include <vector>
#include <stdexcept>
#include <functional>

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
template <typename Content = std::vector<double>>
class RAFKO_EXPORT DataRingbuffer{
public:
  DataRingbuffer(std::uint32_t buffer_number, std::function<void(Content&)> initer)
  :  m_data(buffer_number)
  {
    for(Content& buffer : m_data) initer(buffer);
  }

  /**
   * @brief      Store the current data and move the iterator forward for the next one
   */
  void copy_step(){
    m_currentIndex = (m_currentIndex + 1)%(m_data.size());
    if(1 < m_data.size())
      std::copy(get_element(1).begin(),get_element(1).end(),get_element(0).begin());
  }

  /**
   * @brief      Move the iterator forward for the next data buffer, emptying it
   */
  void clean_step(){
    m_currentIndex = (m_currentIndex + 1)%(m_data.size());
    reset(get_element(0));
  }

  /**
   * @brief      Move the iterator forward and do nothing more
   */
  constexpr void shallow_step(){
    m_currentIndex = (m_currentIndex + 1)%(m_data.size());
  }

  /**
   * @brief      Resets every data element to all zeroes.
   */
  void reset(){
    m_currentIndex =(m_data.size()-1); /* Set the current index into the last index, so at the next @ */
    reset(m_data);
  }

  /**
   * @brief      Removes the first element from the Queue by
   *             filling the latest item with zeroes, and setting the
   *             current index one step back into the past.
   */
  void pop_front(){
    std::fill(get_element(0).begin(),get_element(0).end(),(0.0));
    m_currentIndex = get_buffer_index(1);
  }

  /**
   * @brief      Take over the latest row from the provided buffer
   *
   * @param[in]  other  The buffer to take the data from
   */
  void copy_latest(const DataRingbuffer& other){
    std::copy(
      other.get_element(0).begin(),other.get_element(0).end(), get_element(0).begin()
    );
  }

  /**
   * @brief      Gets the whole o the underlying data as a constant reference
   *
   * @return     The non-modifyable raw buffer data
   */
  constexpr const std::vector<Content>& get_whole_buffer() const{
    return m_data;
  }

  /**
   * @brief      Gets a data value under the provided index parameters
   *
   * @param[in]  past_index  The past index
   * @param[in]  data_index  The index of the data point to retrive in the buffer
   *
   * @return     The value of the data in the   given index parameters
   */
  typename Content::value_type get_element(std::uint32_t past_index, std::uint32_t data_index) const{
    if((m_data.size() > past_index)&&(m_data[0].size() > data_index))
      return get_element(past_index)[data_index];
      else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

  /**
   * @brief      Gets a data value under the provided index parameters
   *
   * @param[in]  past_index  The past index
   * @param[in]  data_index  The index of the data point to retrive in the buffer
   *
   * @return     The value of the data in the   given index parameters
   */
  typename Content::value_type& get_element(std::uint32_t past_index, std::uint32_t data_index){
    if((m_data.size() > past_index)&&(m_data[0].size() > data_index))
      return get_element(past_index)[data_index];
      else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

  /**
   * @brief      Sets the data element under the given indices to the provided value
   *
   * @param[in]  past_index  The buffer to set the data
   * @param[in]  data_index  The index of the data inside the buffer
   * @param[in]  value       The value to overwrite the data with
   */
  void set_element(std::uint32_t past_index, std::uint32_t data_index, typename Content::value_type value){
    if((m_data.size() > past_index)&&(m_data[0].size() > data_index))
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
  Content& get_element(std::uint32_t past_index){
    if(past_index < m_data.size()){
      return m_data[get_buffer_index(past_index)];
    }else throw std::runtime_error("Ringbuffer index out of bounds!");
  }

  /**
   * @brief      Gets a const reference to a stored entry in the ringbuffer
   *
   * @param[in]  past_index  The past index
   *
   * @return     The reference pointing to a data
   */
  const Content& get_element(std::uint32_t past_index) const{
    if(past_index < m_data.size()){
      return m_data[get_buffer_index(past_index)];
    }else throw std::runtime_error("Ringbuffer index out of bounds!");
  }

  /**
   * @brief      Gets the number of buffers stored in the object
   *
   * @return     The sequence size.
   */
  std::uint32_t get_sequence_size() const{
    return m_data.size();
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
  std::int32_t get_sequence_index(std::uint32_t sequence_index, std::uint32_t reach_past_loops) const{
    return (get_sequence_size() - sequence_index - 1) + reach_past_loops;
  }

  /**
   * @brief      Returns the size of the available memory buffers
   *
   * @return     Number of elements
   */
  std::uint32_t buffer_size() const{
    return m_data[0].size();
  }

  /**
   * @brief      Returns the number of available memory buffer
   *
   * @return     number of buffers available
   */
  std::uint32_t buffer_number() const{
    return m_data.size();
  }

  typename Content::value_type& operator[](std::uint32_t index){
    if(index < m_data[0].size())
      return get_element(0, index);
      else throw std::runtime_error("Ringbuffer data index out of bounds!");
  }

private:
  std::uint32_t m_currentIndex = 0u;
  std::vector<Content> m_data;

  /**
   * @brief      Gets the buffer index for the given past index
   *
   * @param[in]  past_index  The past index
   *
   * @return     The buffer index.
   */
  std::uint32_t get_buffer_index(std::uint32_t past_index) const{
    if(m_data.size() <= past_index) throw std::runtime_error("Older data queried, than memory capacity.");
    if(past_index > m_currentIndex) return(m_data.size() + m_currentIndex - past_index);
      else return(m_currentIndex - past_index);
  }

  /**
   * @brief resets different underlying vector types
   */
  constexpr void reset(double buf){
    buf = (0.0);
  }

  void reset(std::vector<double>& buf){
    for(double& element : buf) element = (0.0);
  }

  void reset(std::vector<std::vector<double>>& buf){
    for(std::vector<double>& inner_vector : buf)
      for(double& element : inner_vector)
          element = (0.0);
  }

  void reset(std::vector<std::vector<std::vector<double>>>& buf){
    for(std::vector<std::vector<double>>& outer_vector : buf)
      for(std::vector<double>& inner_vector : outer_vector)
        for(double& element : inner_vector)
            element = (0.0);
  }
};

} /* namespace rafko_utilities */

#endif /* DATA_RINGBUFFER_H */
