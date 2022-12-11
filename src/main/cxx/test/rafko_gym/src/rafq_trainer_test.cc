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

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <string>
#include <iostream>

#include "rafko_net/services/rafko_net_builder.hpp"
#include "rafko_net/services/solution_solver.hpp"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"
#include "rafko_mainframe/services/rafko_gpu_context.hpp"
#else
#include "rafko_mainframe/services/rafko_cpu_context.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_gym/services/rafq_trainer.hpp"
#include "rafko_gym/models/rafko_cost.hpp"

#include "test/test_utility.hpp"

namespace rafko_gym_test {

/**
 * @brief     A collcetion of characters to help level generation for @ConsoleJumper
 */
class CharacterCollection{
public:
  CharacterCollection(std::string collection) : m_collection(collection) { }
  char next() const{ return m_collection[rand()%m_collection.size()]; }
  bool contains(char c){ return m_collection.find(c) != std::string::npos; }
private:
  std::string m_collection;
};

/**
 * @brief     An environment where the player (`o`) tries to jump to the end of the console, 
 *            with boosters(`>`), setbacks(`<`) their extenders (`=`) its path besides 
 *            non-effect characters (`#`) and teleportation pads(`^`). 
 *            Teleportation pads trasnfers the player to the last of it in the console line. 
 *            The player has a limited vision of its surroundings.
 *            - state: energy + what the player sees
 *            - action: where the player moves relative to itself
 */
class ConsoleJumper : public rafko_gym::RafQEnvironment{

  inline static CharacterCollection s_allCharacters = CharacterCollection("####=<>");
  inline static CharacterCollection s_extensions = CharacterCollection("=");
  inline static CharacterCollection s_extensionStoppers = CharacterCollection("<>");

  static std::vector<char> generate_level(std::uint32_t width, std::uint32_t sight, std::uint32_t& last_teleport_position){
    REQUIRE(1 < width);
    std::uint32_t current_extender_count = 0;
    std::uint32_t current_path_count = 0;
    std::vector<char> level(width, '#');
    char prev = level[0];
    for(std::uint32_t char_index = 1; char_index < width; ++char_index){
      if(char_index == (width-1)){
        level[char_index] = '#';
        break;
      }

      if(s_extensionStoppers.contains(prev)){
        current_extender_count = 0;
        prev = '#';
        level[char_index] = '#';
        continue;
      }

      if( (0 < current_path_count) && (rand()%(current_path_count) >= 2) ){
        last_teleport_position = std::max(last_teleport_position, char_index);
        current_path_count = 0;
        prev = '^';
        level[char_index] = '^';
        continue;
      }

      char c;
      if(0 < current_extender_count){
        if(current_extender_count >= (sight/2)){
          current_extender_count = 0;
          c = s_extensionStoppers.next();
        }else c = s_extensions.next();
      }else{
        if(s_extensionStoppers.contains(prev)) c = '#';
          else c = s_allCharacters.next();
      }
      if(c == '=') ++current_extender_count;
      if(c == '#') ++current_path_count;
        else current_path_count = 0;
      prev = c;
      level[char_index] = c;
    }
    return level;
  }

  static double get_q_value(double position, double energy){
    return position + energy / 10.0;
  }

public:
  ConsoleJumper(std::uint32_t width = 80, std::uint32_t sight = 5)
  : rafko_gym::RafQEnvironment((1u + sight), 1, {100, 100}, {0, 7})
  , m_consoleWidth(width)
  , m_sight(sight)
  , m_level(generate_level(m_consoleWidth, m_sight, m_lastTeleportPosition))
  , m_statesBuffer(m_level.size(), std::vector<double>(m_sight))
  , m_actualState(1u + m_sight)
  , m_tmpState(1u + m_sight)
  {
    using LevelView = rafko_utilities::ConstVectorSubrange<std::vector<char>::const_iterator>;

    for(std::uint32_t pos = 0; pos < m_level.size(); ++pos){
      const std::uint32_t start_index = pos - std::min(m_sight, pos);
      const std::uint32_t state_count = std::min(static_cast<std::size_t>(m_sight), (m_level.size() - start_index));
      REQUIRE(start_index + state_count <= m_level.size());
      LevelView level_sight(m_level.begin() + start_index, state_count);
      std::uint32_t start_in_buffer = 0;
      std::uint32_t end_in_buffer = m_sight;
      std::uint32_t start_in_view = 0;
      if(pos >= (m_level.size() - (m_sight/2))){
        end_in_buffer = (m_sight/2) + m_level.size() - pos;
        start_in_view = m_sight - end_in_buffer;
      }else if(pos < (m_sight/2)){
       start_in_buffer = (m_sight/2) - pos;
       start_in_view = 0;
      }

      // if(19 == pos){
      //   std::cout << "pos: " << pos << std::endl; 
      //   std::cout << "m_sight: " << m_sight << std::endl; 
      //   std::cout << "start_index: " << start_index << "/" << m_level.size() << std::endl; 
      //   std::cout << "state_count: " << state_count << std::endl; 
      //   std::cout << "start_in_buffer: " << start_in_buffer << std::endl; 
      //   std::cout << "end_in_buffer: " << end_in_buffer << std::endl; 
      // }
      std::uint32_t buffer_index = 0;
      std::uint32_t view_index = start_in_view;
      std::transform(
        m_statesBuffer[pos].begin(), m_statesBuffer[pos].end(), m_statesBuffer[pos].begin(),
        [&buffer_index, &view_index, &level_sight, start_in_buffer, end_in_buffer](const double&){
          if(
            (buffer_index >= start_in_buffer) && (buffer_index < end_in_buffer)
            &&(view_index < level_sight.size())
          ){
            return static_cast<double>(level_sight[view_index++]);
          }else{ /* Out of bounds elements are padded to zero */
            ++buffer_index;
            return 0.0;
          } 
        }
      );
    }
    reset();
  }

  void print(){
    std::int32_t i = 0; 
    for(char c : m_level){
      if(i++ != m_pos) std::cout << c;
        else std::cout << 'o';
    }
    std::cout << "(" << m_actualState[0] << ")  "; /* the first element of the state is energy */
  }

  std::size_t state_size(){
    return m_actualState.size();
  }

  void reset() override{
    m_pos = 0;
    m_actualState[0] = m_level.size()/2.0;
    std::copy(m_statesBuffer[m_pos].begin(), m_statesBuffer[m_pos].end(), m_actualState.begin() + 1u);
  }

  MaybeFeatureVector current_state() const override{
    if(m_pos < 0 || static_cast<std::int32_t>(m_level.size()) < m_pos){
      return {};
    }
    return m_actualState;
  }

  StateTransition next(FeatureView action) override{
    if(action[0] < -action[0] || m_sight < action[0] || (m_tmpState[0] <= 0.0)) /* if player tries to jump too much it dies*/
      return {{}, static_cast<double>(m_pos), true};

    m_pos += action[0];
    m_actualState[0] -= std::max(0.0, action[0]);
    process(m_level, m_pos, m_lastTeleportPosition);

    double q_value = get_q_value(static_cast<double>(m_pos), m_actualState[0]);
    if(
      (m_pos < 0 || static_cast<std::int32_t>(m_level.size()) <= m_pos)
      ||(m_actualState[0] <= 0.0) /* no energy left */
    ){
      return {{}, q_value , true};
    }
    std::copy(m_statesBuffer[m_pos].begin(), m_statesBuffer[m_pos].end(), m_actualState.begin() + 1);
    return {m_actualState, q_value, false};
  }

  StateTransition next(FeatureView state, FeatureView action) const override{
    REQUIRE(state.size() == m_tmpState.size());
    m_tmpState = state.acquire();
    if(action[0] < -action[0] || m_sight < action[0] || (m_tmpState[0] <= 0.0)) /* if player tries to jump too much it dies*/
      return {{}, (static_cast<double>(m_pos) + m_tmpState[0]), true};

    auto state_it = std::find_if(m_statesBuffer.begin(), m_statesBuffer.end(),
      [&state](const FeatureVector& stored_state){
        std::uint32_t i = 1; /* energy is not compared to stored states */
        REQUIRE(stored_state.size() == (state.size() - 1));
        return std::all_of(
          stored_state.begin(), stored_state.end(),
          [&state, &i](const double& element){ return element == state[i++]; }
        );
      }
    );

    if(state_it == m_statesBuffer.end()){
      return {{}, 0.0, true};
    }

    m_tmpState[0] -= std::max(0.0, action[0]);
    double q_value = get_q_value(static_cast<double>(m_pos), m_tmpState[0]);
    std::int32_t result_index = (state_it - m_statesBuffer.begin()) + action[0];
    process(m_level, result_index, m_lastTeleportPosition);

    if( (result_index < static_cast<std::int32_t>(m_level.size())) && (0 <= result_index) ){
      RFASSERT(result_index < static_cast<std::int32_t>(m_statesBuffer.size()));
      std::copy(m_statesBuffer[result_index].begin(), m_statesBuffer[result_index].end(), m_tmpState.begin() + 1);
      return {{m_tmpState}, q_value, false};
    }else{
      return {{}, q_value, true};
    } 
  }

private:
  const std::uint32_t m_consoleWidth;
  const std::uint32_t m_sight;
  const std::vector<char> m_level;
  std::vector<FeatureVector> m_statesBuffer;
  std::uint32_t m_lastTeleportPosition;
  FeatureVector m_actualState;
  mutable FeatureVector m_tmpState;
  std::int32_t m_pos = 0;

  static void process(const std::vector<char>& level, std::int32_t& pos, std::uint32_t last_teleport_position){
    if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    if('^' == level[pos]){
      pos = last_teleport_position;
      return;
    }

    if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    if('>' == level[pos]){
      ++pos;
    }else if('>' == level[pos]){
      --pos;
    }

    if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    while('=' == level[pos]){
      ++pos;
      if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    }

    std::int32_t direction = 0;
    if('>' == level[pos])
      direction = 1;
      else if('<' == level[pos])direction = -1;

    pos += direction;
    if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    while('=' == level[pos]){
      pos += direction;
      if(pos < 0 || static_cast<std::int32_t>(level.size()) <= pos) return;
    }
  }
};

TEST_CASE("Testing if RafQTrainer works as expected", "[.][ConsoleJumper]") {
  double action;
  ConsoleJumper test_game((rafko_test::get_console_width() - 10), 7);
  test_game.reset();
  while(true){
    std::cout << "\r";
    test_game.print();
    std::cout << ":"; 
    std::cin >> action;
    test_game.next(std::vector<double>{action});
  }
}

TEST_CASE("Testing if RafQTrainer works as expected with a simple board game simulation", "[optimize][QLearning][!benchmark]") {
  constexpr const std::uint32_t policy_action_count = 3;
  constexpr const std::uint32_t policy_action_size = 1;
  constexpr const std::uint32_t policy_sight = 7;
  constexpr const std::uint32_t policy_q_set_size = 500;

  google::protobuf::Arena arena; /* so the network and trainer would be on the same Arena */
  std::shared_ptr<ConsoleJumper> test_game = std::make_shared<ConsoleJumper>(
    (rafko_test::get_console_width()/2), policy_sight
  );
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>(
    rafko_mainframe::RafkoSettings()
    .set_arena_ptr(&arena)
    .set_learning_rate(2e-5).set_minibatch_size(policy_q_set_size / 10).set_memory_truncation(2)
    .set_droput_probability(0.0)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_stop_if_training_error_zero, true)
    .set_training_strategy(rafko_gym::Training_strategy::training_strategy_early_stopping, false)
    .set_learning_rate_decay({{100u,0.8}})
    .set_training_relevant_loop_count(3)
    .set_arena_ptr(&arena).set_max_solve_threads(4).set_max_processing_threads(3)
  );

  rafko_net::RafkoNet& network = *rafko_net::RafkoNetBuilder(*settings)
    .input_size(test_game->state_size()).expected_input_range(test_game->state_properties().m_standardDeviation)
    .add_feature_to_layer(1u, rafko_net::neuron_group_feature_boltzmann_knot)
    .allowed_transfer_functions_by_layer({
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    .create_layers({5,5,(policy_action_count * (1 + policy_action_size))});

  std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
    *settings, rafko_gym::cost_function_mse
  );
  rafko_net::SolutionSolver::Factory solverFactory(network, settings);
  std::shared_ptr<rafko_net::SolutionSolver> reference_solver = solverFactory.build();
  rafko_gym::RafQTrainer trainer(network, policy_action_count, policy_q_set_size, test_game, objective, settings);
  std::uint32_t iteration = 1;
  while(true){
    double exploration_ratio = (1.0 - (static_cast<double>(std::min(500u,iteration))/501));
    bool terminal = false;
    std::uint32_t steps = 0;
    test_game->reset();
    while(!terminal && steps < 200){
      std::cout << "\r" 
      << "opt loops: " << ((exploration_ratio > 0.8)?(0.0):((1.0 - exploration_ratio) * 1000)) << "    "
      << "; iter: " << iteration
      << "; expl: " << exploration_ratio
      << "; qSet size: " << trainer.q_set_size()
      // << "; err: " << trainer.stochastic_evaluation(false/*to_seed*/, 0/*seed_value*/, true/*force_gpu_upload*/) << "; ";
      << "; err: " << trainer.full_evaluation(true/*force_gpu_upload*/) << "; ";
      test_game->print();
      if(!test_game->current_state().has_value()){
        std::cout << "GAME OVER" << std::endl;
        break;
      }
      solverFactory.refresh_actual_solution_weights();
      auto policy_action = reference_solver->solve(test_game->current_state().value(), true/*reset_neuron_data*/);
      auto state_transition = test_game->next({policy_action.begin() + 1, 1});
      terminal = state_transition.m_terminal;
      ++steps;
      std::cout << "|" << policy_action[1] << "  "; 
    }
    // std::cout << std::endl;
    trainer.iterate(
      200/*max_discovery_length*/, 
      exploration_ratio, 
      (exploration_ratio > 0.8)?(0):((1.0 - exploration_ratio) * 1000)/*q_set_training_epochs*/,
      [](double progress){ std::cout << "\r progress: " << (progress * 100) << "%   " << std::flush; }
    );
    ++iteration;
  }
}

} /* namespace rafko_gym_test */
