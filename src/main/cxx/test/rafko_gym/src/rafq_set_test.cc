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

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "rafko_gym/models/rafq_environment.hpp"
#include "rafko_gym/models/rafq_set.hpp"
#include "rafko_mainframe/models/rafko_settings.hpp"

#include "test/test_utility.hpp"

namespace {
rafko_gym::RafQEnvironment::FeatureVector
action_slot(const rafko_gym::RafQEnvironment::FeatureVector &action,
            double q_value) {
  return rafko_gym::RafQSetItemConstView::action_slot(action, q_value);
}

} /* namespace */

namespace rafko_gym_test {

/**
 * @brief     A test environment with 5 internal states, one dead-end local
 * minima ( state 4 ) and a big value state ( 5 ) reached through a low value (
 * 3 ) state only.
 */
class TestEnvironment : public rafko_gym::RafQEnvironment {

  static inline const std::vector<FeatureVector> s_states = {
      {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};

  static inline const std::map<double, bool> s_stateTerminalValues = {
      {1.0, false}, {2.0, false}, {3.0, false}, {4.0, true}, {5.0, false}};

  static inline const std::map<double, double> s_stateQValues = {
      {1.0, 10.0}, {2.0, 20.0}, {3.0, 0.0}, {4.0, 40.0}, {5.0, 666.0}};

  /** @brief description of state transitions: {state, action} --> result state
   */
  static inline const std::map<std::pair<double, double>, double>
      s_stateTransitions = {{{1.0, 2.0}, 2.0},
                            {{1.0, 4.0}, 4.0},
                            {{2.0, 3.0}, 3.0},
                            {{3.0, 5.0}, 5.0},
                            {{5.0, 2.0}, 2.0}};

public:
  TestEnvironment() : rafko_gym::RafQEnvironment(1, 1) {}

  static std::uint32_t states_count() { return s_states.size(); }

  static std::uint32_t max_q_set_size() {
    return s_states.size() - 1u; /* state 4 has no next states, so it won't be
                                    involved in the stored states */
  }

  void reset() override { m_state = s_states[0]; }

  StateTransition current_state() const override {
    return {m_state, s_stateQValues.at(m_state[0]),
            s_stateTerminalValues.at(m_state[0]),
            AnyData(new double(m_state[0]),
                    [](void *ptr) { delete static_cast<double *>(ptr); })};
  }

  StateTransition next(FeatureView action) override {
    StateTransition ret(next(m_state, action));
    if (ret.m_resultState.has_value())
      m_state = ret.m_resultState.value().get();
    return ret;
  }

  StateTransition next(FeatureView state, FeatureView action,
                       const AnyData &user_data = {}) const override {
    REQUIRE(state.size() == state_size());
    REQUIRE(action.size() == action_size());

    if (user_data) {
      REQUIRE(state[0] == *static_cast<double *>(user_data.get()));
    }

    auto result_state = s_stateTransitions.find({state[0], action[0]});
    if (result_state == s_stateTransitions.end()) {
      return {{}, 0.0, true};
    } else {
      auto state_iterator =
          std::find_if(s_states.begin(), s_states.end(),
                       [&result_state](const FeatureVector &stored_state) {
                         return stored_state[0] == std::get<1>(*result_state);
                       });
      return {{*state_iterator},
              s_stateQValues.at((*state_iterator)[0]),
              s_stateTerminalValues.at((*state_iterator)[0]),
              AnyData(new double((*state_iterator)[0]),
                      [](void *ptr) { delete static_cast<double *>(ptr); })};
    }
  }

private:
  FeatureVector m_state = s_states[0];
};

TEMPLATE_TEST_CASE_SIG("Testing if RafQSet element insertion works as expected",
                       "[QSet][QLearning]", ((int ActionCount), ActionCount), 1,
                       2, 3, 4) {
  constexpr const std::uint32_t max_set_size = 4u;

  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_learning_rate(
          1.0); /* learning rate set to 1.0 to make testing TD q values easier
                 */

  TestEnvironment environment;
  rafko_gym::RafQSet q_set(settings, environment, ActionCount, max_set_size,
                           0.1);
  REQUIRE(0 == q_set.get_number_of_sequences());

  /*!Note: in the below comments {x,y} means --> {state,action} */
  SECTION("Adding new actions for the same state") {

    TestEnvironment::StateTransition next_state = environment.current_state();
    REQUIRE(0 == q_set.get_number_of_sequences());

    /* Store the initial value user data, as it is used in incorporate */
    std::vector<rafko_gym::RafQEnvironment::AnyData> user_data;
    std::vector<rafko_gym::RafQEnvironment::AnyData> user_data2;
    user_data.push_back(std::move(next_state.m_userData));
    user_data2.push_back(std::move(next_state.m_userData));

    /* The Q Value of state 2 ( result of {1,2} ) is 20 */
    new (&next_state) TestEnvironment::StateTransition(
        environment.next(std::vector<double>{1.0}, std::vector<double>{2.0}));
    REQUIRE(static_cast<bool>(next_state.m_userData));
    q_set.incorporate({{1.0}}, {action_slot({2.0}, next_state.m_resultQValue)},
                      std::move(user_data));
    REQUIRE(1 == q_set.get_number_of_sequences());

    /* Add a fake, worse qValue for {1,4}; to see if it would get overwritten */
    q_set.incorporate({{1.0}}, {{10.0, 4.0}});
    REQUIRE(1 == q_set.get_number_of_sequences());

    /* The Q Value of state 4 ( result of {1,4} ) is 40 */
    new (&user_data) std::vector<rafko_gym::RafQEnvironment::AnyData>();
    user_data.push_back(std::move(next_state.m_userData));
    new (&next_state) TestEnvironment::StateTransition(
        environment.next(std::vector<double>{1.0}, std::vector<double>{4.0}));
    REQUIRE(static_cast<bool>(next_state.m_userData));
    q_set.incorporate({{1.0}}, {action_slot({4.0}, next_state.m_resultQValue)},
                      std::move(user_data2));
    REQUIRE(1 == q_set.get_number_of_sequences());

    rafko_gym::RafQSetItemView element_view(q_set[0]);
    REQUIRE(element_view.max_q_value() ==
            Catch::Approx(40.0).epsilon(0.0000000000001));
    if (1 < ActionCount) {
      REQUIRE(
          element_view.avg_q_value() ==
          Catch::Approx((20.0 + 40.0) / ActionCount).epsilon(0.0000000000001));
    }
  }

  SECTION("Checking if a worse initial action, which would lead to a better "
          "state gets stored") {
    q_set.incorporate({{1.0}},
                      {action_slot({2.0}, environment
                                              .next(std::vector<double>{1.0},
                                                    std::vector<double>{2.0})
                                              .m_resultQValue)});
    q_set.incorporate({{1.0}},
                      {action_slot({4.0}, environment
                                              .next(std::vector<double>{1.0},
                                                    std::vector<double>{4.0})
                                              .m_resultQValue)});
    REQUIRE(1 == q_set.get_number_of_sequences());

    rafko_gym::RafQSetItemConstView element_view(
        q_set[0]); /* first state is under the first index */
    REQUIRE(
        element_view[0][0] ==
        Catch::Approx(4.0).epsilon(
            0.0000000000001)); /* {1,4} is in the first place in the actions */
    double initial_q_value = element_view.q_value(0);

    q_set.incorporate({{2.0}},
                      {action_slot({3.0}, environment
                                              .next(std::vector<double>{2.0},
                                                    std::vector<double>{3.0})
                                              .m_resultQValue)});
    REQUIRE(2 == q_set.get_number_of_sequences());

    q_set.incorporate({{3.0}},
                      {action_slot({5.0}, environment
                                              .next(std::vector<double>{3.0},
                                                    std::vector<double>{5.0})
                                              .m_resultQValue)});
    REQUIRE(3 == q_set.get_number_of_sequences());

    /*!Note: At this point since the initial states for {1,2} is already stored,
     * re-adding the same state-action pair would include the additional
     * actions, so the qValue of {1,2} wouldbe updated
     */
    new (&element_view) rafko_gym::RafQSetItemConstView(
        q_set[0]); /* Iterator is invalidated so a new object is required */
    q_set.incorporate({{1.0}},
                      {action_slot({2.0}, environment
                                              .next(std::vector<double>{1.0},
                                                    std::vector<double>{2.0})
                                              .m_resultQValue)});
    REQUIRE(element_view[0][0] ==
            Catch::Approx(2.0).epsilon(
                0.0000000000001)); /* {1,2} moved to the first place in the
                                      actions */
    REQUIRE(initial_q_value < element_view.q_value(0));
  }

  SECTION("Checking if the QSet keeps it's maximum size, and the worse states "
          "always get deleted") {
    REQUIRE(0 == q_set.get_number_of_sequences());

    constexpr const std::uint32_t elements_to_upload = 3;
    for (std::uint32_t element_index = 1; element_index <= elements_to_upload;
         ++element_index) {
      q_set.incorporate(
          {
              /* in each iteration upload states with higher q values */
              {static_cast<double>(element_index)},
              {static_cast<double>(element_index * 2)},
              {static_cast<double>(element_index * 3)},
              {static_cast<double>(element_index * 4)},
          },
          {action_slot({4.0} /*action*/,
                       static_cast<double>(element_index + element_index)),
           action_slot({3.0} /*action*/,
                       static_cast<double>(element_index + 2 * element_index)),
           action_slot({2.0} /*action*/,
                       static_cast<double>(element_index + 3 * element_index)),
           action_slot(
               {1.0} /*action*/,
               static_cast<double>(element_index + 4 * element_index))});
    }
    REQUIRE(q_set.get_number_of_sequences() == max_set_size);
    for (std::uint32_t element_index = 0; element_index < q_set.max_size();
         ++element_index) {
      rafko_gym::RafQSetItemConstView element_view(q_set[element_index]);
      /*!Note: Because each iteration added as many elements as the set max
       * size, the last iteration should overwrite the previous ones. Because of
       * this, every state should be >= @elements_to_upload
       */
      REQUIRE(elements_to_upload <= element_view.state()[0]);
      REQUIRE(elements_to_upload < element_view.max_q_value());
    }
    /* check if the best is always kept */
  }

  SECTION("Checking if a currently available state-action pair is updated, the "
          "ordering of the actions are kept according to the q-values") {
    REQUIRE(0 == q_set.get_number_of_sequences());
    q_set.incorporate({{1.0}, {1.0}, {1.0}, {1.0}},
                      {
                          action_slot({1.0} /*action*/, 3.0 /*q_value*/),
                          action_slot({2.0} /*action*/, 4.0 /*q_value*/),
                          action_slot({3.0} /*action*/, 2.0 /*q_value*/),
                          action_slot({4.0} /*action*/, 1.0 /*q_value*/),
                      });
    REQUIRE(1 == q_set.get_number_of_sequences());
    rafko_gym::RafQSetItemConstView element_view(q_set[0]);
    REQUIRE(element_view[0][0] ==
            2.0); /* The action with the best q Value is supposed to be 2.0*/
    REQUIRE(element_view[3][0] ==
            4.0); /* The action with the worst q Value is supposed to be 4.0*/

    /* Update the worst action to be the best */
    q_set.incorporate({{1.0}},
                      {action_slot({4.0} /*action*/, 5.0 /*q_value*/)});
    REQUIRE(1 == q_set.get_number_of_sequences());
    new (&element_view) rafko_gym::RafQSetItemConstView(q_set[0]);
    REQUIRE(element_view[0][0] ==
            4.0); /* The action with the best q Value is supposed to be 4.0*/
  }

  SECTION("Checking the only case when a negative q-value is accepted: When "
          "its state is not already present") {
    REQUIRE(0 == q_set.get_number_of_sequences());

    q_set.incorporate({{1.0}},
                      {action_slot({4.0} /*action*/, -5.0 /*q_value*/)});
    REQUIRE(1 == q_set.get_number_of_sequences());

    rafko_gym::RafQSetItemConstView element_view(q_set[0]);
    REQUIRE(element_view[ActionCount - 1][0] ==
            4.0); /* The action with the worst q Value is supposed to be 4.0 */
    REQUIRE(element_view.min_q_value() == -5.0);
  }
}

TEST_CASE("Testing if RafQSet conversion works as expected",
          "[QSet][QLearning]") {
  using FeatureVector = rafko_gym::RafQEnvironment::FeatureVector;

  constexpr const std::uint32_t action_count = 5u;
  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_learning_rate(
          1.0); /* learning rate set to 1.0 to make testing TD q values easier
                 */

  TestEnvironment environment;
  rafko_gym::RafQSet q_set(settings, environment, action_count,
                           TestEnvironment::max_q_set_size(), 0.1);

  /* Filling up qSet with every posible state and action in the test environment
   * twice! so that correct correct qvalues are stored for the transitively
   * better actions */
  for (std::uint32_t fill = 0; fill < 2; ++fill) {
    for (double state_index = 0; state_index < TestEnvironment::states_count();
         state_index += 1.0) {
      std::vector<FeatureVector> actions_for_state;
      for (double action_index = 0;
           action_index < TestEnvironment::states_count();
           action_index += 1.0) {
        rafko_gym::RafQEnvironment::StateTransition state_transition =
            environment.next(std::vector<double>{state_index + 1.0},
                             std::vector<double>{action_index + 1.0});
        if (state_transition.m_resultState.has_value()) {
          actions_for_state.push_back(
              action_slot(std::vector<double>{action_index + 1.0} /*action*/,
                          state_transition.m_resultQValue));
        }
      }
      q_set.incorporate(
          std::vector<FeatureVector>(actions_for_state.size(),
                                     {state_index + 1.0}) /* states */,
          actions_for_state);
    }
  }

  SECTION("Checking if QSet can build with reduced action count correctly") {
    constexpr const std::uint32_t reduced_action_count =
        action_count - std::min(3u, action_count);
    const std::uint32_t action_slot_size =
        (reduced_action_count *
         rafko_gym::RafQSetItemConstView::action_slot_size(
             environment.action_size()));

    rafko_gym::RafQSet reduced_q_set(q_set, reduced_action_count);
    for (std::uint32_t state_index = 0;
         state_index < q_set.get_number_of_sequences(); ++state_index) {
      REQUIRE_THAT(
          q_set.get_input_sample(state_index),
          Catch::Matchers::Approx(reduced_q_set.get_input_sample(state_index))
              .margin(0.0000000000001));
      FeatureVector label_reference = {
          q_set.get_label_sample(state_index).begin(),
          q_set.get_label_sample(state_index).begin() +
              (reduced_action_count * action_slot_size)};
      REQUIRE_THAT(
          label_reference,
          Catch::Matchers::Approx(reduced_q_set.get_label_sample(state_index))
              .margin(0.0000000000001));
    }
  }

  SECTION("Testing if generating the best actions as sequences working as "
          "expected") {
    REQUIRE(TestEnvironment::max_q_set_size() ==
            q_set.get_number_of_sequences());
    rafko_gym::DataSetPackage generated = q_set.generate_best_sequences(
        4u); /* best sequence is 4 steps long in test env */
    REQUIRE(generated.input_size() == 1u);
    REQUIRE(generated.feature_size() == 1u);
    REQUIRE(generated.sequence_size() == 4u);
    REQUIRE_THAT(
        std::vector<double>(generated.inputs().begin(),
                            generated.inputs().end()),
        Catch::Matchers::Approx(std::vector<double>{1.0, 2.0, 3.0, 5.0})
            .margin(0.0000000000001));
    REQUIRE_THAT(
        std::vector<double>(generated.labels().begin(),
                            generated.labels().end()),
        Catch::Matchers::Approx(std::vector<double>{2.0, 3.0, 5.0, 2.0})
            .margin(0.0000000000001));
  }
}

TEST_CASE("Testing if RafQSet lookup works as expected",
          "[QSet][QLearning][lookup]") {
  constexpr const std::uint32_t max_set_size = 4u;
  constexpr const std::uint32_t action_count = 2u;

  rafko_mainframe::RafkoSettings settings =
      rafko_mainframe::RafkoSettings().set_learning_rate(
          1.0); /* learning rate set to 1.0 to make testing TD q values easier
                 */

  /* Testing if looking up separate states work as expected */
  TestEnvironment environment;
  rafko_gym::RafQSet q_set(settings, environment, action_count, max_set_size,
                           0.1);
  REQUIRE(0 == q_set.get_number_of_sequences());
  q_set.incorporate({{1.0}, {2.0}, {3.0}, {4.0}},
                    {
                        action_slot({1.0} /*action*/, 3.0 /*q_value*/),
                        action_slot({2.0} /*action*/, 4.0 /*q_value*/),
                        action_slot({3.0} /*action*/, 2.0 /*q_value*/),
                        action_slot({4.0} /*action*/, 1.0 /*q_value*/),
                    });

  std::uint32_t test_index;
  REQUIRE(q_set.look_up(std::vector<double>{1.0}).has_value());
  REQUIRE(
      1.0 ==
      q_set.look_up(std::vector<double>{1.0}, &test_index).value().get()[0]);
  REQUIRE(test_index < max_set_size);

  REQUIRE(q_set.look_up(std::vector<double>{2.0}).has_value());
  REQUIRE(
      2.0 ==
      q_set.look_up(std::vector<double>{2.0}, &test_index).value().get()[0]);
  REQUIRE(test_index < max_set_size);

  REQUIRE(q_set.look_up(std::vector<double>{3.0}).has_value());
  REQUIRE(
      3.0 ==
      q_set.look_up(std::vector<double>{3.0}, &test_index).value().get()[0]);
  REQUIRE(test_index < max_set_size);

  REQUIRE(q_set.look_up(std::vector<double>{4.0}).has_value());
  REQUIRE(
      4.0 ==
      q_set.look_up(std::vector<double>{4.0}, &test_index).value().get()[0]);
  REQUIRE(test_index < max_set_size);

  settings.set_delta(0.25); /* Setting delta determines the difference to state
                               value where it is considered different */
  /*!Note: Since The cost function Mean Squared error is used: a difference of
   * 0.5 will result in the error (label - (label + 0.5))^2 = 0.5^2 = 0.25.
   * Setting delta to 0.3 will eliminate the possiblity for rounding errors.
   */
  rafko_gym::RafQSet::MaybeFeatureVector query_result =
      q_set.look_up(std::vector<double>{1.5});
  double queried_state = query_result.value().get()[0];
  REQUIRE((queried_state == 1.0 || queried_state == 2.0));

  query_result = q_set.look_up(std::vector<double>{3.5});
  queried_state = query_result.value().get()[0];
  REQUIRE((queried_state == 3.0 || queried_state == 4.0));
}

} /* namespace rafko_gym_test */
