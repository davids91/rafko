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
#include <map>

#include "rafko_net/services/synapse_iterator.hpp"
#include "rafko_protocol/rafko_net.pb.h"

#include "test/test_utility.hpp"

namespace rafko_net_test {

// TODO: Testcase for synpase index start

TEST_CASE("Testing Interval start index inside a synapse",
          "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::IndexSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::uint32_t>> synapse_indexes = {
      {50, 10}, {60, 30}, {20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_weights() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<> iter(neuron.input_weights());
  REQUIRE(iter.interval_starts_at(0) == 0);
  REQUIRE(iter.interval_starts_at(1) == 10);
  REQUIRE(iter.interval_starts_at(2) == 40);
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the
 * laid out indexes
 */
TEST_CASE("Synapse Iteration", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::uint32_t>> synapse_indexes = {
      {50, 10}, {60, 30}, {20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  std::uint32_t range_iter = 0;
  std::int32_t manual_index = synapse_indexes[0][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());

  REQUIRE(110 == iter.size());

  iter.iterate([&](std::int32_t index) {
    REQUIRE(synapse_indexes.size() > range_iter);
    CHECK(index == manual_index);
    ++manual_index;
    if (((synapse_indexes.size() - 1) >
         range_iter) /* Only adapt range iter, if there is a next range element
                      */
        && (manual_index - synapse_indexes[range_iter][0]) >=
               synapse_indexes[range_iter][1]) {
      ++range_iter;
      manual_index = synapse_indexes[range_iter][0];
    }
  });
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the
 * laid out indexes, based on ranges
 */
TEST_CASE("Synapse iteration on a range", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::uint32_t>> synapse_indexes = {
      {50, 10}, {60, 30}, {20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  std::uint32_t range_iter = 1;
  std::int32_t manual_index = synapse_indexes[1][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());

  REQUIRE(110 == iter.size());

  iter.iterate(
      [&](std::int32_t index) {
        REQUIRE(synapse_indexes.size() > range_iter);
        CHECK(index == manual_index);
        ++manual_index;
        if (((synapse_indexes.size() - 1) >
             range_iter) /* Only adapt range iter, if there is a next range
                            element */
            && (manual_index - synapse_indexes[range_iter][0]) >=
                   synapse_indexes[range_iter][1]) {
          ++range_iter;
          manual_index = synapse_indexes[range_iter][0];
        }
      },
      1, 2);
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if the indexes follow the
 * laid out indexes, even with negative numbers
 */
TEST_CASE("Synapse iteration including negative numbers",
          "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::int32_t>> synapse_indexes = {
      {-50, 10}, {-60, 30}, {-20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  std::uint32_t range_iter = 0;
  std::int32_t manual_index = synapse_indexes[0][0];
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());

  REQUIRE(110 == iter.size());

  iter.iterate([&](std::int32_t index) {
    REQUIRE(synapse_indexes.size() > range_iter);
    CHECK(index == manual_index);
    --manual_index;
    if (((synapse_indexes.size() - 1) >
         range_iter) /* Only adapt range iter, if there is a next range element
                      */
        && (std::abs(manual_index - synapse_indexes[range_iter][0]) >=
            synapse_indexes[range_iter][1])) {
      ++range_iter;
      manual_index = synapse_indexes[range_iter][0];
    }
  });
}

/*###############################################################################################
 * Testing synapse iteration
 * - Creating an artificial synapse pair, and testing if operator[] is reaching
 * the correct indexes and correctly mapping the synapse inputs into a contigous
 * array
 */
TEST_CASE("Synapse Iterator direct access", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::int32_t>> synapse_indexes = {
      {-50, 10}, {70, 30}, {-20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());
  CHECK(iter[0] == -50);
  CHECK(iter[5] == -55);
  CHECK(iter[10] == 70);
  CHECK(iter[11] == 71);
  CHECK(iter[12] == 72);
  CHECK(iter[39] == 99);
  CHECK(iter[40] == -20);
  CHECK(iter[109] == -89);
}

/*###############################################################################################
 * Testing synapse Skimming
 * - Creating an artificial synapse pair, and testing if skim operation goes
 * through all the synapses, correctly displaying starting indices and sizes
 */
TEST_CASE("Synapse Iterator Skimming", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::int32_t>> synapse_indexes = {
      {-50, 10}, {70, 30}, {-20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());

  std::int32_t manual_index = 0;
  iter.skim([&](rafko_net::InputSynapseInterval input_synapse) {
    CHECK(input_synapse.starts() == synapse_indexes[manual_index][0]);
    CHECK(static_cast<std::int32_t>(input_synapse.interval_size()) ==
          synapse_indexes[manual_index][1]);
    ++manual_index;
  });
}

/*###############################################################################################
 * Testing synapse utility functions
 * - Creating an artificial synapse pair, testing .size and .back
 */
TEST_CASE("Synapse Iterator Utility functions", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::InputSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::int32_t>> synapse_indexes = {
      {-50, 10}, {70, 30}, {-20, 70}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_indices() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iter(
      neuron.input_indices());
  CHECK(110 == iter.size());
  CHECK(-89 == iter.back());
}

/*###############################################################################################
 * Testing if synapse iteration is valid with given ranges as well
 * - Create an artificial synapse array of at least 3 elements
 * - See if the iteration goes correctly for each
 */
TEST_CASE("Ranged Synapse iteration", "[synapse-iteration]") {
  rafko_net::Neuron neuron = rafko_net::Neuron();
  rafko_net::IndexSynapseInterval temp_synapse_interval;
  std::vector<std::vector<std::int32_t>> synapse_indexes = {
      {50, 3},
      {70, 3},
      {20, 2},
      {30, 2}}; /* {{range},{start,length},{range}..} */

  for (std::uint32_t i = 0; i < synapse_indexes.size(); ++i) {
    temp_synapse_interval.set_starts(synapse_indexes[i][0]);
    temp_synapse_interval.set_interval_size(synapse_indexes[i][1]);
    *neuron.add_input_weights() = temp_synapse_interval;
  }

  rafko_net::SynapseIterator<> iter(neuron.input_weights());
  std::uint32_t iterate_range_number;
  std::int32_t iteration_count;
  std::uint32_t current_synapse;
  std::int32_t element_index;

  for (std::uint32_t i = 0; i < synapse_indexes.size();
       ++i) { /* go through each synapse interval */
    element_index = synapse_indexes[i][0]; /* the start of the synapse currently
                                              being iterated over */
    iteration_count = 0;
    iter.iterate(
        [&](std::int32_t synapse_index) {
          CHECK(element_index == synapse_index);
          ++element_index;
          ++iteration_count;
        },
        i, 1);

    CHECK(iteration_count == synapse_indexes[i][1]);
  }

  std::int32_t iteration_in_this_synapse;
  for (std::uint32_t i = 0; i < synapse_indexes.size();
       ++i) { /* go through each synapse interval */
    current_synapse = i;
    iterate_range_number =
        std::min(2u, static_cast<std::uint32_t>(synapse_indexes.size() - i));
    element_index = synapse_indexes[i][0]; /* the start of the synapse currently
                                              being iterated over */
    iteration_count = 0;
    iteration_in_this_synapse = 0;
    iter.iterate(
        [&](std::int32_t synapse_index) {
          if (iteration_in_this_synapse >=
              synapse_indexes[current_synapse][1]) {
            ++current_synapse;
            iteration_in_this_synapse = 0;
            element_index = synapse_indexes[current_synapse][0];
          }

          CHECK(element_index == synapse_index);

          ++iteration_in_this_synapse;
          ++element_index;
          ++iteration_count;
        },
        i, iterate_range_number);

    for (std::uint32_t j = 0; j < iterate_range_number; ++j) {
      iteration_count -= synapse_indexes[i + j][1];
    }

    CHECK(0 == iteration_count);
  }
}

/*###############################################################################################
 * Testing if synapse iterator equality operator produces correct output
 */
TEST_CASE("Synapseiterator equality", "[synapse-iteration]") {
  rafko_net::Neuron neuron1 = rafko_net::Neuron();
  rafko_net::Neuron neuron2 = rafko_net::Neuron();
  rafko_net::Neuron neuron3 = rafko_net::Neuron();
  std::uint32_t num_synapses = (rand() % 50) + 1;
  for (std::uint32_t synapse_index = 0; synapse_index < num_synapses;
       ++synapse_index) {
    rafko_net::IndexSynapseInterval temp_interval;
    temp_interval.set_starts(rand() % 435);
    temp_interval.set_interval_size(rand() % 435);
    *neuron1.add_input_weights() = temp_interval;
    *neuron2.add_input_weights() = temp_interval;
    temp_interval.set_starts(500u + rand() % 435);
    temp_interval.set_interval_size(200u + rand() % 435);
    *neuron3.add_input_weights() = temp_interval;
  }

  CHECK(rafko_net::SynapseIterator<>(neuron1.input_weights()) ==
        rafko_net::SynapseIterator<>(neuron2.input_weights()));
  CHECK(rafko_net::SynapseIterator<>(neuron1.input_weights()) !=
        rafko_net::SynapseIterator<>(neuron3.input_weights()));
  CHECK(rafko_net::SynapseIterator<>(neuron2.input_weights()) !=
        rafko_net::SynapseIterator<>(neuron3.input_weights()));

  num_synapses = (rand() % 50) + 1;
  for (std::uint32_t synapse_index = 0; synapse_index < num_synapses;
       ++synapse_index) {
    rafko_net::IndexSynapseInterval temp_interval;
    temp_interval.set_starts(rand() % 435);
    temp_interval.set_interval_size(rand() % 435);
    *neuron1.add_input_weights() = temp_interval;
    temp_interval.set_starts(500u + rand() % 435);
    temp_interval.set_interval_size(200u + rand() % 435);
    *neuron2.add_input_weights() = temp_interval;
    *neuron3.add_input_weights() = temp_interval;
  }

  CHECK(rafko_net::SynapseIterator<>(neuron1.input_weights()) !=
        rafko_net::SynapseIterator<>(neuron2.input_weights()));
  CHECK(rafko_net::SynapseIterator<>(neuron1.input_weights()) !=
        rafko_net::SynapseIterator<>(neuron3.input_weights()));
  CHECK(rafko_net::SynapseIterator<>(neuron2.input_weights()) !=
        rafko_net::SynapseIterator<>(neuron3.input_weights()));
}

TEST_CASE("Testing Utility functions reach_back_loops and interval_size_of of "
          "SynapseIterator",
          "[synapse-iteration]") {
  std::vector<std::size_t> synapse_sizes(rand() % 5u + 1u);
  std::vector<std::uint32_t> synapse_reachbacks(synapse_sizes.size());
  std::uint32_t overall_elements = 0u;
  google::protobuf::RepeatedPtrField<rafko_net::InputSynapseInterval> synapses;
  std::map<std::uint32_t, std::uint32_t> nth_to_size;
  std::map<std::uint32_t, std::uint32_t> nth_to_reachback;
  for (std::uint32_t synapse_index = 0; synapse_index < synapse_sizes.size();
       ++synapse_index) {
    synapse_sizes[synapse_index] = rand() % 100u;
    synapse_reachbacks[synapse_index] = rand() % 10u;
    synapses.Add()->set_interval_size(synapse_sizes[synapse_index]);
    synapses.Mutable(synapse_index)->set_starts(overall_elements);
    synapses.Mutable(synapse_index)
        ->set_reach_past_loops(synapse_reachbacks[synapse_index]);
    for (std::uint32_t n = 0; n < synapse_sizes[synapse_index]; ++n) {
      nth_to_size.insert({overall_elements + n, synapse_sizes[synapse_index]});
      nth_to_reachback.insert(
          {overall_elements + n, synapse_reachbacks[synapse_index]});
    }
    overall_elements += synapse_sizes[synapse_index];
  }
  rafko_net::SynapseIterator<rafko_net::InputSynapseInterval> iterator(
      synapses);
  for (std::uint32_t variant = 0u; variant < 100u; ++variant) {
    std::uint32_t index = rand() % overall_elements;
    REQUIRE(iterator.interval_size_of(index) == nth_to_size[index]);
    REQUIRE(iterator.reach_past_loops<rafko_net::InputSynapseInterval>(index) ==
            nth_to_reachback[index]);
  }
}

} // namespace rafko_net_test
