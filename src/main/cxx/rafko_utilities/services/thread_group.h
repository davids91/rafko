/*! This file is part of davids91/Rafko.
 *
 *   Rafko is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Rafko is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *   <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#ifndef THREAD_GROUP_H
#define THREAD_GROUP_H

#include "rafko_global.h"
#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <atomic>
#include <condition_variable>
#include <cassert>

namespace rafko_utilities{

/**
 * @brief    This class provides a number of worker threads to be executed in paralell for the functionality
 *          defined in the constructor of the template object. The class itself is not thread safe! Use a mutex
 *          to run the same instance from multiple threads.
 */
class RAFKO_FULL_EXPORT ThreadGroup{
public:
  ThreadGroup(std::uint32_t number_of_threads){
    assert(0u < number_of_threads);
    for(std::uint32_t i = 0; i < number_of_threads; ++i)
     threads.emplace_back(std::thread(&ThreadGroup::worker, this, i));
  }

  ~ThreadGroup(){
    { /* Signal to the worker threads that the show is over */
     std::lock_guard<std::mutex> my_lock(state_mutex);
     state.store(End);
    }
    while(0 < threads.size()){
      synchroniser.notify_all();
      if(threads.back().joinable()){
        threads.back().join();
        threads.pop_back();
      }
    }
  }

  void start_and_block(const std::function<void(std::uint32_t)>& function) const;

  /**
   * @brief     Returns the number of worker threads handled in this group
   */
  std::uint32_t get_number_of_threads() const{
    return threads.size();
  }

private:
  enum state_t{Idle, Start, End};
  mutable const std::function<void(std::uint32_t)>* worker_function; /* gets the thread index it is inside */
  mutable std::size_t threads_ready = 0;
  mutable std::atomic<state_t> state = {Idle};
  mutable std::mutex state_mutex;
  mutable std::condition_variable synchroniser;
  std::vector<std::thread> threads;

  void worker(std::uint32_t thread_index);
};

} /* namespace rafko_utilities */
#endif /* THREAD_GROUP_H */
