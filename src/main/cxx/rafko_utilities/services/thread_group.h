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
#include <tuple>
#include <vector>
#include <thread>
#include <mutex>
#include <iomanip>
#include <numeric>
#include <atomic>
#include <condition_variable>

namespace rafko_utilities{

using std::atomic;
using std::vector;
using std::function;
using std::tuple;
using std::thread;
using std::mutex;
using std::unique_lock;
using std::lock_guard;
using std::condition_variable;
using std::size_t;

/**
 * @brief    This class provides a number of worker threads to be executed in paralell for the functionality
 *          defined in the constructor of the template object.
 */
class ThreadGroup{
public:
  ThreadGroup(uint32 number_of_threads)
  {
    for(uint32 i = 0; i < number_of_threads; ++i)
     threads.emplace_back(thread(&ThreadGroup::worker, this, i));
  }

  ~ThreadGroup(){
    { /* Signal to the worker threads that the show is over */
     lock_guard<mutex> my_lock(state_mutex);
     state.store(End);
    }
    synchroniser.notify_all();
    for(thread& thread : threads) thread.join();
  }

  void start_and_block(function<void(uint32)>& function){
    { /* initialize, start.. */
     unique_lock<mutex> my_lock(state_mutex);
     worker_function = function;
     state.store(Start);
    }
    synchroniser.notify_all(); /* Whip the peons */

    { /* wait until the work is done */
     unique_lock<mutex> my_lock(state_mutex);
     synchroniser.wait(my_lock,[this](){
      return (threads.size() <= threads_ready);
     });
    }
    { /* set appropriate state */
     unique_lock<mutex> my_lock(state_mutex);
     state.store(Idle);
    }
    synchroniser.notify_all(); /* Notify worker threads that the main thread is finished */

    { /* wait until all threads are notified */
     unique_lock<mutex> my_lock(state_mutex);
     synchroniser.wait(my_lock,[this](){
      return (0 >= threads_ready); /* All threads are notified once the @threads_ready variable is zero again */
     });
    }
  }

private:
  enum state_t{Idle, Start, End};

  function<void(uint32)> worker_function; /* gets the thread index it is inside */
  vector<thread> threads;
  size_t threads_ready = 0;
  atomic<state_t> state = {Idle};
  mutex state_mutex;
  condition_variable synchroniser;

  void worker(uint32 thread_index){
    while(End != state.load()){ /* Until the pool is stopped */
     { /* Wait until main thread triggers a task */
      unique_lock<mutex> my_lock(state_mutex);
      synchroniser.wait(my_lock,[this](){
        return (Idle != state.load());
      });
     }
     if(End != state.load()){ /* In case there are still tasks to execute.. */
       worker_function(thread_index);/* do the work */

       { /* signal that work is done! */
        unique_lock<mutex> my_lock(state_mutex);
        ++threads_ready; /* increase "done counter" */
       }
       synchroniser.notify_all(); /* Notify main thread that this thread  is finsished */

       { /* Wait until main thread is closing the iteration */
        unique_lock<mutex> my_lock(state_mutex);
        synchroniser.wait(my_lock,[this](){
          return (Start != state.load());
        });
       }

       { /* signal that this thread is notified! */
        unique_lock<mutex> my_lock(state_mutex);
        --threads_ready; /* decrease the "done counter" to do so */
       }
       synchroniser.notify_all(); /* Notify main thread that this thread  is finsished */
     }
    } /*while(END_VALUE != state)*/
  }
};

} /* namespace rafko_utilities */
#endif /* THREAD_GROUP_H */
