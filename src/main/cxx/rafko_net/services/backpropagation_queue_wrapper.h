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

#ifndef BACKPROPAGATION_WRAPPER_H
#define BACKPROPAGATION_WRAPPER_H

#include "rafko_mainframe/models/service_context.h"

namespace rafko_net{

using rafko_mainframe::Service_context;

/**
 * @brief      Wrapper function to generate Backpropagation_queue objects from @RafkoNet
 *             objects
 */
class Backpropagation_queue_wrapper{
public:
  Backpropagation_queue_wrapper(RafkoNet& net, Service_context& context);
  Backpropagation_queue operator()(){
    return gradient_step;
  }
private:
  Backpropagation_queue gradient_step;
};

} /* namespace rafko_net */

#endif /* BACKPROPAGATION_WRAPPER_H */
