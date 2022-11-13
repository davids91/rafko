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

#include <vector>
#include <utility>
#include <algorithm>
#include <map>

#include "rafko_mainframe/models/rafko_settings.hpp"
#include "rafko_gym/models/rafq_set.hpp"
#include "rafko_gym/models/rafq_environment.hpp"

#include "test/test_utility.hpp"

namespace rafko_mainframe_test {

class TestEnvironment : public rafko_gym::RafQEnvironment{

  static inline const std::vector<DataType> s_states = {
    {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
  };

  static inline const std::map<double, bool> s_stateTerminalValues = {
    {1.0, false}, {2.0, false}, {3.0, false}, {4.0, true}, {5.0, false}
  };  

  static inline const std::map<double, double> s_stateQValues = {
    {1.0, 10.0}, {2.0, 20.0}, {3.0, 0.0}, {4.0, 30.0}, {5.0, 666.0}
  };

  /** @brief description of state transitions: {state, action} --> result state
   */
  static inline const std::map<std::pair<double, double>, double> s_stateTransitions = {
    {{1.0, 2.0}, 2.0},
    {{1.0, 4.0}, 4.0},
    {{2.0, 3.0}, 3.0},
    {{3.0, 5.0}, 5.0},
    {{5.0, 2.0}, 2.0}
  };

public:
  TestEnvironment()
  : rafko_gym::RafQEnvironment(1,1)
  , m_dummyBuffer()
  {
  }

  StateTransition next(const DataType& state, const DataType& action){
    REQUIRE(state.size() == state_size());
    REQUIRE(action.size() == action_size());

    auto result_state = s_stateTransitions.find({state[0], action[0]});
    if(result_state == s_stateTransitions.end())
      return {{}, 0.0, true};
    else{
      auto state_iterator = std::find_if(
        s_states.begin(), s_states.end(),
        [&result_state](const DataType& stored_state){ return stored_state[0] == std::get<1>(*result_state); }
      );
      return {
        {*state_iterator}, 
        s_stateQValues.at((*state_iterator)[0]), 
        s_stateTerminalValues.at((*state_iterator)[0])
      };
    }   
  }
private:
  const DataType m_dummyBuffer;
};

TEST_CASE("Testing if RafQSet works as expected", "[QSet][QLearning]") {
  /*
   test cases:
   directly better action, 
   worse action indirectly way better( to test recursive state transitions ), 
   set size check-->worse gets deleted,
   best action is always first
  */
  constexpr const std::uint32_t max_set_size = 4u;
  rafko_mainframe::RafkoSettings settings;
  TestEnvironment environment;
  rafko_gym::RafQSet<4> q_set(settings, environment, max_set_size, 0.1);

  REQUIRE(0 == q_set.get_number_of_sequences());
  q_set.incorporate({{1.0}}, {{1.0, 2.0}});
  REQUIRE(1 == q_set.get_number_of_sequences());

  q_set.incorporate({{1.0}}, {{5.0, 3.0}});
  REQUIRE(1 == q_set.get_number_of_sequences());

  q_set.incorporate({{1.0}}, {{10.0, 3.0}});
  REQUIRE(1 == q_set.get_number_of_sequences());

  rafko_gym::RafQSetItemView<4> element_view(q_set[0]);
  REQUIRE( element_view.max_q_value() == Catch::Approx(10.0).epsilon(0.0000000000001) );
  REQUIRE( element_view.avg_q_value() == Catch::Approx(7.5).epsilon(0.0000000000001) );

  //TODO: get incorporated data from environment
}

} /* namespace rafko_mainframe_test */
