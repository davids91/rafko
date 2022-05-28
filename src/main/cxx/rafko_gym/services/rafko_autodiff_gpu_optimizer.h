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

#ifndef RAFKO_AUTODIFF_GPU_OPTIMIZER_H
#define RAFKO_AUTODIFF_GPU_OPTIMIZER_H

#include "rafko_global.h"

#include <cmath>
#include <memory>
#include <CL/opencl.hpp>

#include "rafko_mainframe/models/rafko_gpu_strategy_phase.h"
#include "rafko_mainframe/services/rafko_gpu_phase.h"
#include "rafko_gym/services/rafko_autodiff_gpu_strategy.h"
#include "rafko_gym/services/rafko_autodiff_optimizer.h"

namespace rafko_gym{

/**
 * @brief
 */
class RAFKO_FULL_EXPORT RafkoAutodiffGPUOptimizer
: private RafkoAutodiffOptimizer
{
public: //TODO: Initialize OCL context
  RafkoAutodiffGPUOptimizer(
    const rafko_mainframe::RafkoSettings& settings_,
    std::shared_ptr<RafkoEnvironment> environment_, rafko_net::RafkoNet& network_,
    std::shared_ptr<rafko_mainframe::RafkoContext> training_evaluator_ = {},
    std::shared_ptr<rafko_mainframe::RafkoContext> test_evaluator_ = {}
  )
  : RafkoAutodiffOptimizer(settings_, environment_, network_, training_evaluator_, test_evaluator_)
  , strategy(std::make_shared<AutoDiffGPUStrategy>(settings, network))
  // , gpu_phase()
  {
    strategy->set_environment(environment);
  }

  void build(std::shared_ptr<RafkoObjective> objective){
    std::uint32_t weight_relevant_operation_count = build_without_data(objective);
    strategy->build(operations);
    //TODO: Sort operations into groups that can be executed in paralell
    // |--> traverse operations backwards, collecting everything solvable
    // |--> collect them, and mark them solved
    // |--> traverse again, collect newly solvable operations
    // |--> collect them, and mark them solved
    // |--> repeat until there are operations left
    //TODO: build phase  
    // gpu_phase.set_strategy(strategy);
  }

  void iterate(){
    //TODO: Update stochastic data in buffers
    //TODO: Run Phase
    //TODO: Weight updates based on result data
    //TODO: Training and test set evaluation ?
  }

private:
  std::shared_ptr<AutoDiffGPUStrategy> strategy;
  // rafko_mainframe::RafkoGPUPhase gpu_phase;
};

} /* namespace rafko_gym */

#endif /* RAFKO_AUTODIFF_OCL_OPTIMIZER_H */
