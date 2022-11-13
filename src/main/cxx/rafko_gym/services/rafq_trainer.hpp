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

#ifndef COST_FUNCTION_MSE_H
#define COST_FUNCTION_MSE_H

#include "rafko_global.hpp"

#include <memory>
#include <vector>

#include "rafko_mainframe/rafko_autonomous_entity.hpp"
#include "rafko_mainframe/models.rafko_settings.hpp"
#include "rafko_net/services/solution_solver.hpp"

#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafq_set.hpp"

namespace rafko_gym{

/**
 * @brief      Trainer facilitating QLearning 
 */
template<std::size_t ActionCount>
class RAFKO_EXPORT RafQTrainer : public RafkoAutonomousEntity{
  using DataType = RafQSet<ActionCount>::DataType;
  using ExperienceEntry = std::pair<std::vector<DataType>, std::vector<DataType>>;
public:
  RafQTrainer(
    rafko_net::RafkoNet& network, std::uint32_t q_set_size, std::shared_ptr<RafQEnvironment> environment = {}, 
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings = {}
  )
  : RafkoAutonomousEntity(settings)
  , m_network(network)
  , m_solverFactory(m_network, m_settings)
  , m_agent(m_solverFactory.build())
  , m_environment(environment)
  , m_qSet(???)
  , m_optimizer(???)
  {
  }

  void set_environment(std::shared_ptr<RafQEnvironment> environment){
    RFASSERT(static_cast<bool>(environment));
    m_environment = environment;
    //TODO: re-init QSet and optimizer
  }

  void iterate(){
    //TODO: run until terminal or length
    //TODO: collect state-action-qvalue pairs from env
    //TODO: update q-set
    //TODO: train agent from qset
    //TODO: parameterize run(count) and training(epoch) in one iteration
  }

private:
  rafko_net::RafkoNet& m_network;
  rafko_net::SolutionSolver::Factory m_solverFactory;
  rafko_net::SolutionSolver m_agent;
  std::shared_ptr<RafQEnvironment> m_environment;
  RafQSet<ActionCount> m_qSet;
  std::vector<ExperienceEntry> m_xpDataBase;
  std::unique_ptr<rafko_gym::RafkoAutodiffOptimizer> m_optimizer;
};

} /* namespace rafko_gym */

#endif /* COST_FUNCTION_MSE_H */
