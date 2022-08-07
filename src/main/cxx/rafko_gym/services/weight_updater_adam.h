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

#ifndef WEIGHT_UPDATER_ADAM_H
#define WEIGHT_UPDATER_ADAM_H

#include "rafko_gym/services/rafko_weight_updater.h"

#include <vector>

namespace rafko_gym {

class RAFKO_FULL_EXPORT RafkoWeightUpdaterAdam : public RafkoWeightUpdater{
public:
  RafkoWeightUpdaterAdam(rafko_net::RafkoNet& rafko_net, const rafko_mainframe::RafkoSettings& settings_)
  :  RafkoWeightUpdater(rafko_net, settings_)
  ,  moment(rafko_net.weight_table_size(),(0.0))
  ,  raw_moment(rafko_net.weight_table_size(),(0.0))
  { }

  void iterate(const std::vector<double>& gradients) override;

protected:
  double get_new_velocity(std::uint32_t weight_index, const std::vector<double>& gradients) const override;

private:
  std::uint32_t iteration_count = 0u;
  std::vector<double> moment;
  std::vector<double> raw_moment;
};

} /* namespace rafko_gym */

#endif /* WEIGHT_UPDATER_ADAM_H */
