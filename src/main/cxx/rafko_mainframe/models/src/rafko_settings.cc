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

#include "rafko_mainframe/models/rafko_settings.h"


namespace rafko_mainframe{

double RafkoSettings::get_learning_rate(std::uint32_t iteration) const{
  if((0 == learning_rate_with_decay.size())||(iteration < std::get<std::uint32_t>(learning_rate_with_decay[0])))
    return hypers.learning_rate();
  if(iteration >= std::get<std::uint32_t>(learning_rate_with_decay.back()))
    return std::get<double>(learning_rate_with_decay.back());
  std::uint32_t decay_index = 0;
  if(iteration >= learning_rate_decay_iteration_cache)
    decay_index = learning_rate_decay_index_cache;

  while(
    (decay_index < (learning_rate_with_decay.size()-1u))
    &&(iteration >= std::get<std::uint32_t>(learning_rate_with_decay[decay_index]))
  )++decay_index;

  --decay_index;

  learning_rate_decay_iteration_cache = iteration;
  learning_rate_decay_index_cache = decay_index;

  return std::get<double>(learning_rate_with_decay[decay_index]);
}

} /* rafko_mainframe */
