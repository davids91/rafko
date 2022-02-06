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

#ifndef DATA_POOL_H
#define DATA_POOL_H

#include "rafko_global.h"

#include <vector>
#include <deque>
#include <mutex>
#include <algorithm>

namespace rafko_utilities{

/**
 * @brief      This data container allocates buffers on-demand and handles
 *              data acess. The container is thread-safe.
 */
template<class T = sdouble32>
class RAFKO_FULL_EXPORT DataPool{
public:
  DataPool(uint32 pool_size, uint32 expected_buffer_size)
  : buffer_pool(pool_size, std::vector<T>())
  {
    std::for_each(buffer_pool.begin(),buffer_pool.end(),[=](std::vector<T>& buf){
      buf.reserve(expected_buffer_size);
    });
  }

  DataPool() = default;

  std::vector<T>& reserve_buffer(uint32 number_of_elements){
    std::lock_guard<std::mutex> my_lock(buffers_mutex);
  	for(uint32 buffer_index = 0; buffer_index < buffer_pool.size();++buffer_index){
  		if(0u == buffer_pool[buffer_index].size()){ /* if the vector[buffer_index] has 0 elements --> the vector is free */
  			buffer_pool[buffer_index].resize(number_of_elements); /* reserve the vector */
  			return buffer_pool[buffer_index]; /* and make it available */
  		}
  	}
  	buffer_pool.push_back(std::vector<T>(number_of_elements));
  	return buffer_pool.back();
  }

  void release_buffer(std::vector<T>& buffer){
  	buffer.resize(0);
  }

private:
  std::deque<std::vector<T>> buffer_pool;
  std::mutex buffers_mutex;
};

} /* namespace rafko_utilities */
#endif /* DATA_POOL_H */
