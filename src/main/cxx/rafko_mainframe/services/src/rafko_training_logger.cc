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

#include "rafko_mainframe/services/rafko_training_logger.hpp"

#include <iostream>
#include <fstream>

namespace rafko_mainframe{

void RafkoTrainingLogger::log(std::uint32_t iteration, const std::vector<std::uint32_t>& coordinates, const std::vector<std::string>& tags, const std::vector<double>& data){
  DataPackage measured;
  measured.set_iteration(iteration);
  for(const std::uint32_t& coordinate : coordinates) measured.add_coordinates(coordinate);
  for(const std::string& tag : tags) measured.add_tags(tag);
  for(const double& data_element : data) measured.add_data(data_element);
  *measurement.add_packs() = measured;
  ++changes_since;
  if(settings.get_tolerance_loop_value() < changes_since)
    flush();
}

void RafkoTrainingLogger::flush(){
  std::filebuf logfile;
  logfile.open(id+".log", std::ios::out | std::ios::binary | std::ios::trunc);
  std::ostream log_stream(&logfile);
  measurement.SerializeToOstream(&log_stream);
  changes_since = 0;
  logfile.close();
}

} /* rafko_mainframe */
