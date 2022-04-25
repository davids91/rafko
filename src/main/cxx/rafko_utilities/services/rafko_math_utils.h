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

#ifndef RAFKO_MATH_UTILS_H
#define RAFKO_MATH_UTILS_H

#include "rafko_global.h"

#include <utility>

namespace rafko_utilities{

/**
 * @brief   Provides a unique hash for a provided pair based on value
 *
 * @param[in]   input   the pair to generate the hash for
 *
 * @return    the generated hash
 */
static constexpr std::uint64_t pair_hash(std::pair<std::int32_t,std::uint32_t> input){
  return(static_cast<std::uint64_t>(std::get<0>(input))|(static_cast<std::uint64_t>(std::get<1>(input)) << 32u));
}


} /* namespace rafko_utilities */

#endif/* RAFKO_MATH_UTILS_H */
