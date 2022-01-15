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

#ifndef RAFKO_NDARRAY_SHAPE
#define RAFKO_NDARRAY_SHAPE

#include "rafko_global.h"

#include <vector>

namespace rafko_mainframe{

/**
 * @brief      A phase of the Deep learning GPU pipeline consisting of several ordered GPU Kernels.
 */
class RAFKO_FULL_EXPORT RafkoNDArrayShape : public std::vector<uint32>{
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_NDARRAY_SHAPE */
