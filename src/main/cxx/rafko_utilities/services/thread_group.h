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
#include <iomanip>
#include <numeric>
#include <atomic>
#include <condition_variable>
#include <assert.h>

namespace rafko_utilities{

/**
 * @brief    This class provides a number of worker threads to be executed in paralell for the functionality
 *          defined in the constructor of the template object. The class itself is not thread safe! Use a mutex
 *          to run the same instance from multiple threads.
 */
class RAFKO_FULL_EXPORT ThreadGroup{
public:
  ThreadGroup(uint32 number_of_threads){
    assert(0u < number_of_threads);
    for(uint32 i = 0; i < number_of_threads; ++i)
     threads.emplace_back(std::thread(&ThreadGroup::worker, this, i));
  }

  ~ThreadGroup(){
    { /* Signal to the worker threads that the show is over */
     std::lock_guard<std::mutex> my_lock(state_mutex);
     state.store(End);
    }
    synchroniser.notify_all();
    for(std::thread& thread : threads) thread.join();
  }

  void start_and_block(const std::function<void(uint32)>& function) const{
    { /* initialize, start.. */
     std::unique_lock<std::mutex> my_lock(state_mutex);
     worker_function = &function;
     state.store(Start);
    }
    synchroniser.notify_all(); /* Whip the peons */

    { /* wait until the work is done */
     std::unique_lock<std::mutex> my_lock(state_mutex);
     synchroniser.wait(my_lock,[this](){
      return (threads.size() <= threads_ready);
     });
    }
    { /* set appropriate state */
     std::unique_lock<std::mutex> my_lock(state_mutex);
     state.store(Idle);
    }
    synchroniser.notify_all(); /* Notify worker threads that the main thread is finished */

    { /* wait until all threads are notified */
     std::unique_lock<std::mutex> my_lock(state_mutex);
     synchroniser.wait(my_lock,[this](){
      return (0u == threads_ready); /* All threads are notified once the @threads_ready variable is zero again */
     });
    }
  }

  /**
   * @brief     Returns the number of worker threads handled in this group
   */
  uint32 get_number_of_threads() const{
    return threads.size();
  }

private:
  enum state_t{Idle, Start, End};
  mutable const std::function<void(uint32)>* worker_function; /* gets the thread index it is inside */
  mutable std::size_t threads_ready = 0;
  mutable std::atomic<state_t> state = {Idle};
  mutable std::mutex state_mutex;
  mutable std::condition_variable synchroniser;
  std::vector<std::thread> threads;

  void worker(uint32 thread_index){
    while(End != state.load()){ /* Until the pool is stopped */
     { /* Wait until main thread triggers a task */
      std::unique_lock<std::mutex> my_lock(state_mutex);
      synchroniser.wait(my_lock,[this](){
        return (Idle != state.load());
      });
     }
     if(End != state.load()){ /* In case there are still tasks to execute.. */
       (*worker_function)(thread_index);/* do the work */

       { /* signal that work is done! */
        std::unique_lock<std::mutex> my_lock(state_mutex);
        ++threads_ready; /* increase "done counter" */
       }
       synchroniser.notify_all(); /* Notify main thread that this thread  is finsished */

       { /* Wait until main thread is closing the iteration */
        std::unique_lock<std::mutex> my_lock(state_mutex);
        synchroniser.wait(my_lock,[this](){
          return (Start != state.load());
        });
       }

       { /* signal that this thread is notified! */
        std::unique_lock<std::mutex> my_lock(state_mutex);
        --threads_ready; /* decrease the "done counter" to do so */
       }
       synchroniser.notify_all(); /* Notify main thread that this thread  is finsished */
     }
    } /*while(END_VALUE != state)*/
  }
};

} /* namespace rafko_utilities */
#endif /* THREAD_GROUP_H */
