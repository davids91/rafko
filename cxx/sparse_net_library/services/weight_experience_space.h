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

#ifndef WEIGHT_EXPERIENCE_SPACE_H
#define WEIGHT_EXPERIENCE_SPACE_H

#include "sparse_net_global.h"

#include <vector>

namespace sparse_net_library{

using std::vector;

class Weight_experience_space{
public:
  Weight_experience_space(sdouble32 weight_min_, sdouble32 weight_max_, sdouble32 weight_step_);

  /**
   * @brief      Adds a positive or negative experience value for the given weight.
   *
   * @param[in]  value  A value signaling an experience. If it's positive, it correlates to fitness,
   *                    if negaitve it correlates to an error value
   *
   * @return     Returns with the value of the weight which has the best experience
   */
  sdouble32 add_experience(sdouble32 value);

  /**
   * @brief      Returns with the value of the weight with the best experience
   *
   * @return     the value of the weight with the best experience
   */
  sdouble32 get_best_weight(void);

private:
  sdouble32 weight_min, weight_max, weight_step;
  vector<sdouble32> weight_values;
  vector<sdouble32> experiences;
  uint32 best_weight_index;
  uint32 smallest_experience;

  /**
   * @brief      Updates @best_weight_index based on the @experiences vector
   */
  void find_best_weight(void);

  /**
   * @brief      Cuts the experience vector with the value of its smallest absolute experience,
   *             to avoid overflow with new experiences
   */
  void cut(void);

};

} /* namespace sparse_net_library */

#endif /*  WEIGHT_EXPERIENCE_SPACE_H */