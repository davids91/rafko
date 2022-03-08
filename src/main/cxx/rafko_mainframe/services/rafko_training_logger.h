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

#ifndef RAFKO_TRAINING_LOGGER_H
#define RAFKO_TRAINING_LOGGER_H

#include "rafko_global.h"

#include <vector>
#include <string>

#include "rafko_protocol/logger.pb.h"

#include "rafko_mainframe/models/rafko_settings.h"

namespace rafko_mainframe{

/**
 * @brief      This class is a helper utility to create measurements about the neuron activations and experiences
 *             during training
 */
class RafkoTrainingLogger{
public:
  RafkoTrainingLogger(std::string id_, RafkoSettings& settings)
  : id(id_)
  , settings(settings)
  { }

  void log(std::uint32_t iteration, const std::vector<std::uint32_t>& coordinates, const std::vector<std::string>& tags, const std::vector<double>& data);
  void flush();

private:
  std::string id;
  RafkoSettings& settings;
  Measurement measurement;
  std::uint32_t changes_since = 0u;
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_TRAINING_LOGGER_H */
