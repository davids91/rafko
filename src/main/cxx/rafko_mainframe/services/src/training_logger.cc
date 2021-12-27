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

#include "rafko_mainframe/services/training_logger.h"

#include <iostream>
#include <fstream>

namespace rafko_mainframe{

void Training_logger::log(uint32 iteration, const std::vector<uint32>& coordinates, const std::vector<std::string>& tags, const std::vector<sdouble32>& data){
  DataPackage measured;
  measured.set_iteration(iteration);
  for(const uint32& coordinate : coordinates) measured.add_coordinates(coordinate);
  for(const std::string& tag : tags) measured.add_tags(tag);
  for(const sdouble32& data_element : data) measured.add_data(data_element);
  *measurement.add_packs() = measured;
  ++changes_since;
  if(settings.get_tolerance_loop_value() < changes_since)
    flush();
}

void Training_logger::flush(){
  std::filebuf logfile;
  logfile.open(id+".log", std::ios::out | std::ios::binary | std::ios::trunc);
  std::ostream log_stream(&logfile);
  measurement.SerializeToOstream(&log_stream);
  changes_since = 0;
  logfile.close();
}

} /* rafko_mainframe */
