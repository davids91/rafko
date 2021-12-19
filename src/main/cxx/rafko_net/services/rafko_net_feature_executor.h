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

#ifndef RAFKO_NETWORK_FEATURE_H
#define RAFKO_NETWORK_FEATURE_H

#include "rafko_global.h"

#include <assert.h>
#include <vector>
#include <memory>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_net/models/rafko_softmax_feature.h"

namespace rafko_net{

/**
 * @brief      A base class for all RafkoNeuralNetwork related features
 */
class RafkoNetFeatureExecutor{
public:
  RafkoNetFeatureExecutor(std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads_)
  : execution_threads(execution_threads_)
  { }

  void execute(const FeatureGroup& feature, std::vector<sdouble32>& neuron_data, uint32 thread_index = 0) const{
    assert(thread_index < execution_threads.size());
    switch(feature.feature()){
      case neuron_group_feature_softmax: RafkoSoftmaxFeature::calculate(neuron_data, feature.relevant_neurons(), *execution_threads[thread_index]);
      default: break;
    }
  }

private:
  std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>>& execution_threads;
};

} /* namespace rafko_net */

#endif /* RAFKO_NETWORK_FEATURE_H */
