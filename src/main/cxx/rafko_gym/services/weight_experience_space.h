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

#include "rafko_global.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace rafko_gym{

using std::vector;

/**
 * @brief      This class describes an experience space for a single weight inside a Neural network.
 *             Experiences can be positive or negative, all of which are stored inside the space in the @experiences
 *             vector. The Weight space strives to always focus on the weight value with the best experience point,
 *             while also remembering negative experiences. The experience values are stored in a relative manner,
 *             as to avoid value overflow: Whenever an experience is added into the weight value in focus, the experience of
 *             the smallest cardinality is set back to zero, and all other experiences are corrected for that.
 */
class WeightExperienceSpace{
public:
  WeightExperienceSpace(sdouble32 weight_min_, sdouble32 weight_max_, sdouble32 weight_step_);

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
  sdouble32 get_best_weight();

  /**
   * @brief      Gets the weights stored in the range of the space
   *
   * @return     A const reference of the vector containing the weight values corresponding with stored experiences.
   */
  const vector<sdouble32> get_weights() const{
    return weight_values;
  }

  /**
   * @brief      Gets a single weight from the space under the provided index
   *
   * @param[in]  index  The index
   *
   * @return     The weight.
   */
  sdouble32 get_weight(uint32 index) const{
    if(weight_values.size() > index)
      return weight_values[index];
    else throw std::runtime_error("Weight index out of bounds in weight experience space!");
  }

  /**
   * @brief      Gets the weight which was the best before the current one
   *
   * @return     The value of the previous best weight.
   */
  sdouble32 get_last_weight() const{
    return weight_values[last_weight_index];
  }

  /**
   * @brief      Gets the value of the left neighbour of best weight.
   *
   * @return     The left neighbour of best weight, in case of the first and lest element, the weight itself is returned.
   */
  sdouble32 get_left_neighbour_of_best() const{
    return weight_values[std::max(1u,best_weight_index)-1];
  }

  /**
   * @brief      Gets the value of the right neighbour of best weight.
   *
   * @return     The right neighbour of best weight, in case of the first and lest element, the weight itself is returned.
   */
  sdouble32 get_right_neighbour_of_best() const{
    return weight_values[std::min(static_cast<uint32>(weight_values.size())-2u,best_weight_index)+1];
  }

  /**
   * @brief      Gets the weight experiences.
   *
   * @return     A constant reference of the vector of the experience values corresponding to each stored weight.
   */
  const vector<sdouble32> get_weight_experiences() const{
    return experiences;
  }

private:
  sdouble32 weight_min, weight_max, weight_step;
  vector<sdouble32> weight_values;
  vector<sdouble32> experiences;
  uint32 best_weight_index;
  uint32 worst_weight_index;
  uint32 last_weight_index;
  uint32 smallest_experience;

  /**
   * @brief      Updates @best_weight_index and @worst_weight_index based on the @experiences vector
   */
  void evaluate_weights();

  /**
   * @brief      Pushes the given weight in the direction of its neightbours based on the experience values
   *
   * @param[in]  weight_index  The weight index; must be inbetween 0 < @weight_index < @weight_values.size()
   */
  void adapt_weight(uint32 weight_index);

  /**
   * @brief      Cuts the experience vector with the value of its smallest absolute experience,
   *             to avoid overflow with new experiences
   */
  void cut();

};

} /* namespace rafko_gym */

#endif /*  WEIGHT_EXPERIENCE_SPACE_H */
