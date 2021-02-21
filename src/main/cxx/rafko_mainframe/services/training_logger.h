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

#ifndef TRAINING_LOGGER_H
#define TRAINING_LOGGER_H

#include "rafko_global.h"

#include <vector>
#include <string>

#include "gen/logger.pb.h"

#include "rafko_mainframe/models/service_context.h"

namespace rafko_mainframe{

using std::string;
using std::vector;

/**
 * @brief      This class is a helper utility to create measurements about the neuron activations and experiences
 *             during training
 */
class Training_logger{
public:
  Training_logger(string id_, Service_context& service_context)
  :  id(id_)
  ,  context(service_context)
  ,  measurement()
  ,  changes_since()
  { }

  void log(uint32 iteration,  vector<uint32> coordinates, vector<string> tags, vector<sdouble32> data);
  void flush();

private:
  string id;
  Service_context& context;
  Measurement measurement;
  uint32 changes_since;
};

} /* namespace rafko_mainframe */

#endif /* TRAINING_LOGGER_H */
