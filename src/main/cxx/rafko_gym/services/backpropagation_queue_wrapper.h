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

#include "rafko_protocol/rafko_net.pb.h"

#include "rafko_mainframe/models/rafko_settings.h"

namespace rafko_gym{

/**
 * @brief      Wrapper function to generate BackpropagationQueue objects from @RafkoNet
 *             objects
 */
class BackpropagationQueueWrapper{
public:
  BackpropagationQueueWrapper(rafko_net::RafkoNet& net, rafko_mainframe::RafkoSettings& settings);
  BackpropagationQueue operator()(){
    return gradient_step;
  }
private:
  BackpropagationQueue gradient_step;
};

} /* namespace rafko_gym */

#endif /* BACKPROPAGATION_WRAPPER_H */
