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

#include "rafko_gym/services/rafko_weight_updater.hpp"

#include <set>

#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym {

void RafkoWeightUpdater::iterate(const std::vector<double>& gradients){
  calculate_velocity(gradients);
  update_weights_with_velocity();
  m_iteration = (m_iteration + 1) % m_requiredIterationsForStep;
  m_finished = (0u == m_iteration);
}

void RafkoWeightUpdater::update_weight_with_velocity(std::uint32_t weight_index, std::uint32_t weight_number){
  for(std::uint32_t weight_iterator = 0; weight_iterator < weight_number; ++weight_iterator){
    m_network.set_weight_table( weight_index + weight_iterator, get_new_weight(weight_index + weight_iterator) );
  }
}

void RafkoWeightUpdater::calculate_velocity(const std::vector<double>& gradients){
  execution_threads.start_and_block([this, &gradients](std::uint32_t thread_index){
    const std::uint32_t weight_index_start = m_weightsToDoInOneThread * thread_index;
    const std::uint32_t weights_to_do_in_this_thread = std::min(
      m_weightsToDoInOneThread,
      static_cast<std::uint32_t>(m_network.weight_table_size() - std::min(m_network.weight_table_size(), static_cast<std::int32_t>(weight_index_start)))
    );
    for(std::uint32_t weight_iterator = 0; weight_iterator < weights_to_do_in_this_thread; ++weight_iterator){
      m_currentVelocity[weight_index_start + weight_iterator] = get_new_velocity(weight_index_start + weight_iterator, gradients);
    }
  });
}

void RafkoWeightUpdater::update_weights_with_velocity(){
  std::lock_guard<std::mutex> my_lock(reference_mutex);
  execution_threads.start_and_block([this](std::uint32_t thread_index){
    std::int32_t weight_index_start = m_weightsToDoInOneThread * thread_index;
    if(weight_index_start < m_network.weight_table_size()){
      std::uint32_t weight_index = (m_weightsToDoInOneThread * thread_index);
      update_weight_with_velocity(weight_index, std::min(m_weightsToDoInOneThread, (m_network.weight_table_size() - weight_index)));
    }
  });

}

} /* namespace rafko_gym */
