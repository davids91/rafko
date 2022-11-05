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
#ifndef RAFKO_CONFIG_H
#define RAFKO_CONFIG_H

#cmakedefine RAFKO_USES_OPENCL @RAFKO_USES_OPENCL@
#cmakedefine CL_HPP_MINIMUM_OPENCL_VERSION @CL_HPP_MINIMUM_OPENCL_VERSION@
#cmakedefine CL_HPP_TARGET_OPENCL_VERSION @CL_HPP_TARGET_OPENCL_VERSION@

#cmakedefine RAFKO_USES_ASSERTLOGS @RAFKO_USES_ASSERTLOGS@

#endif /*RAFKO_CONFIG_H*/