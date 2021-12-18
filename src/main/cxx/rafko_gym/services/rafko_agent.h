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

#ifndef RAFKO_AGENT_H
#define RAFKO_AGENT_H

#include "rafko_global.h"

#include <vector>
#include <functional>

#include "rafko_protocol/solution.pb.h"
#include "rafko_utilities/models/data_ringbuffer.h"
#include "rafko_utilities/models/data_pool.h"
#include "rafko_utilities/models/const_vector_subrange.h"

namespace rafko_gym{

using std::vector;

/**
 * @brief      This class serves as a base for reinforcement learning agent, which provides output data
 *              based on different inputs
 */
class RAFKO_FULL_EXPORT RafkoAgent{
public:
  RafkoAgent(const rafko_net::Solution& brain_, uint32 required_temp_data_size_, uint32 required_temp_data_number_per_thread_, uint32 max_threads_ = 1)
  : brain(brain_)
  , required_temp_data_number_per_thread(required_temp_data_number_per_thread_)
  , required_temp_data_size(required_temp_data_size_)
  , max_threads(max_threads_)
  , common_data_pool((required_temp_data_number_per_thread * max_threads_), required_temp_data_size_)
  , neuron_value_buffers(max_threads, rafko_utilities::DataRingbuffer( brain.network_memory_length(), brain.neuron_number()))
  { /* A temporary buffer is allocated for every required future usage per thread */
    for(uint32 tmp_data_index = 0; tmp_data_index < (required_temp_data_number_per_thread * max_threads); ++tmp_data_index)
      used_data_buffers.push_back(common_data_pool.reserve_buffer(required_temp_data_size));
  }

  /**
   * @brief      Solves the rafko_net::Solution provided in the constructor, previous neural information is supposedly available in @output buffer
   *
   * @param[in]      input    The input data to be taken
   * @return         The output values of the network result
   */
  rafko_utilities::ConstVectorSubrange<> solve(const vector<sdouble32>& input, bool reset_neuron_data, uint32 thread_index = 0){
    if(max_threads > thread_index){
      assert( input.size() == brain.network_input_size() );
      if(reset_neuron_data)neuron_value_buffers[thread_index].reset();
      solve( input, neuron_value_buffers[thread_index], used_data_buffers, (thread_index * required_temp_data_number_per_thread), thread_index );
      return { /* return with the range of the output Neurons */
        neuron_value_buffers[thread_index].get_const_element(0).end() - brain.output_neuron_number(),
        neuron_value_buffers[thread_index].get_const_element(0).end()
      };
    } else throw std::runtime_error("Thread index out of bounds!");
  }

  /**
   * @brief      Solves the rafko_net::Solution provided in the constructor, previous neural information is supposedly available in @output buffer
   *
   * @param[in]      input                    The input data to be taken
   * @param          output                   The used Output data to write the results to
   * @param[in]      tmp_data_pool            The already allocated data pool to be used to store intermediate data
   * @param[in]      used_data_pool_start     The first index inside @tmp_data_pool to be used
   */
  virtual void solve(
    const vector<sdouble32>& input, rafko_utilities::DataRingbuffer& output,
    const vector<std::reference_wrapper<vector<sdouble32>>>& tmp_data_pool,
    uint32 used_data_pool_start = 0, uint32 thread_index = 0
  ) const = 0;

  /**
   * @brief      Provide the underlying solution the solver is built to solve.
   *
   * @return     A const reference to the solution the agent is using to produce outputs to the given inputs
   */
  const rafko_net::Solution& get_solution() const{
    return brain;
  }

  /**
   * @brief      Provide the raw Neural data
   *
   * @param[in]      thread_index     The index of the target thread
   * @return         A const reference to the raw Neuron data
   */
  const rafko_utilities::DataRingbuffer& get_memory(uint32 thread_index) const{
    assert(thread_index < neuron_value_buffers.size());
    return neuron_value_buffers[thread_index];
  }

  virtual ~RafkoAgent() = default;

  /**
   * @brief     Provides the size of the buffer it was declared with
   */
  uint32 get_required_temp_data_size(){
    return required_temp_data_size;
  }

private:
  const rafko_net::Solution& brain;
  uint32 required_temp_data_number_per_thread;
  uint32 required_temp_data_size;
  uint32 max_threads;

  rafko_utilities::DataPool<sdouble32> common_data_pool;
  vector<rafko_utilities::DataRingbuffer> neuron_value_buffers; /* One rafko_utilities::DataRingbuffer per thread */
  vector<std::reference_wrapper<vector<sdouble32>>> used_data_buffers;
};

} /* namespace rafko_gym */
#endif /* RAFKO_AGENT_H */
