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

#include "rafko_utilities/services/thread_group.h"

namespace rafko_utilities {

void ThreadGroup::start_and_block(const std::function<void(std::uint32_t)>& function) const{
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

void ThreadGroup::worker(std::uint32_t thread_index){
  while(End != state.load()){ /* Until the pool is stopped */
    { /* Wait until main thread triggers a task */
      std::unique_lock<std::mutex> my_lock(state_mutex);
      synchroniser.wait(my_lock,[this](){
        return (Idle != state.load());
      });
    }
    if(End != state.load()){ /* In case there are still tasks to execute.. */
      { /* signal that work is done! */
        std::unique_lock<std::mutex> my_lock(state_mutex);
      }
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

} /* namespace rafko_utilities */
