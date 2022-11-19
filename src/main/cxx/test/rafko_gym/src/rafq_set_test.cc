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
    {1.0, 10.0}, {2.0, 20.0}, {3.0, 0.0}, {4.0, 40.0}, {5.0, 666.0}
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

  StateTransition next(const DataType& state, const DataType& action) override{
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
  constexpr const std::uint32_t max_set_size = 4u;
  constexpr const std::uint32_t action_count = 2;

  using XPSlot = rafko_gym::RafQSetItemConstView<1>;
  using QSetSlot = rafko_gym::RafQSetItemConstView<action_count>;

  rafko_mainframe::RafkoSettings settings;
  TestEnvironment environment;
  rafko_gym::RafQSet<action_count> q_set(settings, environment, max_set_size, 0.1);
  REQUIRE(0 == q_set.get_number_of_sequences());

  /*!Note: in the below comments {x,y} means --> {state,action} */
  SECTION("Adding new actions for the same state"){
    /* The Q Value of state 2 ( result of {1,2} ) is 20 */
    q_set.incorporate(
      {{1.0}},
      {XPSlot::action_slot({2.0}/*action*/, environment.next({1.0}, {2.0}).m_resultQValue)}
    );
    REQUIRE(1 == q_set.get_number_of_sequences());

    /* Add a fake, worse qValue for {1,4}; to see if it would get overwritten */
    q_set.incorporate({{1.0}}, {{10.0, 4.0}});
    REQUIRE(1 == q_set.get_number_of_sequences());

    /* The Q Value of state 4 ( result of {1,4} ) is 40 */
    q_set.incorporate(
      {{1.0}},
      {XPSlot::action_slot({4.0}/*action*/, environment.next({1.0}, {4.0}).m_resultQValue)}
    );
    REQUIRE(1 == q_set.get_number_of_sequences());

    rafko_gym::RafQSetItemView<action_count> element_view(q_set[0]);
    REQUIRE( element_view.max_q_value() == Catch::Approx(40.0).epsilon(0.0000000000001) );

    /* average: (20 + 40)/2 == 30 */
    REQUIRE( element_view.avg_q_value() == Catch::Approx(30.0).epsilon(0.0000000000001) );
  }

  SECTION("Checking if a worse initial action, which would lead to a better state gets stored"){
    q_set.incorporate(
      {{1.0}},
      {XPSlot::action_slot({2.0}/*action*/, environment.next({1.0}, {2.0}).m_resultQValue)}
    );
    q_set.incorporate(
      {{1.0}},
      {XPSlot::action_slot({4.0}/*action*/, environment.next({1.0}, {4.0}).m_resultQValue)}
    );
    REQUIRE(1 == q_set.get_number_of_sequences());

    QSetSlot element_view(q_set[0]); /* first state is under the first index */
    REQUIRE(element_view[0][0] == Catch::Approx(4.0).epsilon(0.0000000000001)); /* {1,4} is in the first place in the actions */

    q_set.incorporate(
      {{2.0}},
      {XPSlot::action_slot({3.0}/*action*/, environment.next({2.0}, {3.0}).m_resultQValue)}
    );
    REQUIRE(2 == q_set.get_number_of_sequences());

    q_set.incorporate(
      {{3.0}},
      {XPSlot::action_slot({5.0}/*action*/, environment.next({3.0}, {5.0}).m_resultQValue)}
    );
    REQUIRE(3 == q_set.get_number_of_sequences());

    /*!Note: At this point since the initial states for {1,2} is already stored, re-adding the same 
     * state-action pair would include the additional states, so the qValue of {1,2} would increase
     */
    new (&element_view) QSetSlot(q_set[0]); /* Iterator is invalidated so a new object is required */
    double initial_q_value = element_view.q_value(1); /* the second action is the worse currently */
    q_set.incorporate(
      {{1.0}},
      {XPSlot::action_slot({2.0}/*action*/, environment.next({1.0}, {2.0}).m_resultQValue)}
    );
    REQUIRE(element_view[0][0] == Catch::Approx(2.0).epsilon(0.0000000000001)); /* {1,2} moved to the first place in the actions */
    REQUIRE(initial_q_value < element_view.q_value(0));
  }

  SECTION("Checking if the QSet keeps it's maximum size, and the worse states always get deleted"){
    REQUIRE(0 == q_set.get_number_of_sequences());

    constexpr const std::uint32_t elements_to_upload = 3;
    for(std::uint32_t element_index = 1; element_index <= elements_to_upload; ++element_index){
      q_set.incorporate(
        { /* in each iteration upload states with higher q values */
          {static_cast<double>(element_index)},
          {static_cast<double>(element_index*2)},
          {static_cast<double>(element_index*3)},
          {static_cast<double>(element_index*4)},
        },
        {
          XPSlot::action_slot({4.0}/*action*/, static_cast<double>(element_index + element_index)),
          XPSlot::action_slot({3.0}/*action*/, static_cast<double>(element_index + 2 * element_index)),
          XPSlot::action_slot({2.0}/*action*/, static_cast<double>(element_index + 3 * element_index)),
          XPSlot::action_slot({1.0}/*action*/, static_cast<double>(element_index + 4 * element_index))
        }
      );   
    }
    REQUIRE(q_set.get_number_of_sequences() == max_set_size);
    for(std::uint32_t element_index = 0; element_index < q_set.max_size(); ++element_index){
      QSetSlot element_view(q_set[element_index]);
      /*!Note: Because each iteration added as many elements as the set max size,
       * the last iteration should overwrite the previous ones. Because of this, 
       * every state should be >= @elements_to_upload
       */
      REQUIRE( elements_to_upload <= element_view.state()[0] );
      REQUIRE( elements_to_upload < element_view.max_q_value() );
    }
    /* check if the best is always kept */
  }
}

} /* namespace rafko_mainframe_test */
