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

#ifndef SYNAPSE_ITERATOR_H
#define SYNAPSE_ITERATOR_H

#include "rafko_global.hpp"

#include <functional>
#include <stdexcept>

#include <google/protobuf/repeated_field.h>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_protocol/rafko_net.pb.h"

namespace rafko_net {

/**
 * @brief      This class describes a synapse iterator. Based on the given
 * references it provides a hook to go through every index described by them.
 *             Since synapse iteration is a repeating occurence in Neuron
 * routing, partial detail solution and in Solution input collection as well,
 * this class aims to provide a standardized solution for it.
 *             @param[in]  do_for_each_synapse  A lambda function to process the
 * synapses in each start of the synapse iteration. Lambda parameter is the
 * Synapse size It shall return true if the iteration can continue.
 *
 *             @param[in]  do_for_each_index  A lambda function to process the
 * indexes in each step of the synapse iteration. Lambda parameter is the
 * current Synapse index It shall return true if the iteration can continue.
 *
 *             @param[in] interval_start  Defines the range the iteration shall
 * visit
 *
 *             @param[in] interval_start  Defines the range the iteration shall
 * visit
 *
 *             Default range is the whole of the synapses and the
 * @do_for_each_synapse lambda is optional. Example: To iterate through a
 * synapse set, a lambda for each synapse start, and for each element in that
 * synapse: syn_iter.iterate([&](int synapse_start){},[&](int index){});
 */
template <typename Interval_type = IndexSynapseInterval>
class RAFKO_EXPORT SynapseIterator {
public:
  constexpr SynapseIterator(
      const google::protobuf::RepeatedPtrField<Interval_type>
          &arg_synapse_interval)
      : m_synapseInterval(arg_synapse_interval), m_cachedSizeVar(size()){};

  constexpr void iterate(std::function<void(std::int32_t)> do_for_each_index,
                         std::uint32_t interval_start = 0,
                         std::uint32_t interval_size_ = 0) const {
    iterate(m_synapseInterval, do_for_each_index, interval_start,
            interval_size_);
  }
  constexpr void iterate(std::function<void(Interval_type)> do_for_each_synapse,
                         std::function<void(std::int32_t)> do_for_each_index,
                         std::uint32_t interval_start = 0,
                         std::uint32_t interval_size_ = 0) const {
    iterate(m_synapseInterval, do_for_each_synapse, do_for_each_index,
            interval_start, interval_size_);
  }
  constexpr void
  iterate_terminatable(std::function<bool(std::int32_t)> do_for_each_index,
                       std::uint32_t interval_start = 0,
                       std::uint32_t interval_size_ = 0) const {
    iterate_terminatable(m_synapseInterval, do_for_each_index, interval_start,
                         interval_size_);
  }
  constexpr void
  iterate_terminatable(std::function<bool(Interval_type)> do_for_each_synapse,
                       std::function<bool(std::int32_t)> do_for_each_index,
                       std::uint32_t interval_start = 0,
                       std::uint32_t interval_size_ = 0) const {
    iterate_terminatable(m_synapseInterval, do_for_each_synapse,
                         do_for_each_index, interval_start, interval_size_);
  }
  constexpr void skim(std::function<void(Interval_type)> do_for_each_synapse,
                      std::uint32_t interval_start = 0,
                      std::uint32_t interval_size_ = 0) const {
    skim(m_synapseInterval, do_for_each_synapse, interval_start,
         interval_size_);
  }
  constexpr void
  skim_terminatable(std::function<bool(Interval_type)> do_for_each_synapse,
                    std::uint32_t interval_start = 0,
                    std::uint32_t interval_size_ = 0) const {
    skim_terminatable(m_synapseInterval, do_for_each_synapse, interval_start,
                      interval_size_);
  }

  static void skim(const google::protobuf::RepeatedPtrField<Interval_type>
                       &arg_synapse_interval,
                   std::function<void(Interval_type)> do_for_each_synapse,
                   std::uint32_t interval_start = 0,
                   std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter) {
      do_for_each_synapse(arg_synapse_interval[syn_iter]);
    }
  }

  static void iterate(const google::protobuf::RepeatedPtrField<Interval_type>
                          &arg_synapse_interval,
                      std::function<void(std::int32_t)> do_for_each_index,
                      std::uint32_t interval_start = 0,
                      std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter) {
      std::uint32_t this_interval_size =
          arg_synapse_interval[syn_iter].interval_size();
      std::uint32_t this_interval_start =
          arg_synapse_interval[syn_iter].starts();
      if (!is_index_input(this_interval_start)) {
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator) {
          do_for_each_index(this_interval_start + input_iterator);
        }
      } else /* current @starts. element is from the input, iterate in a
                negative way */
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator) {
          do_for_each_index(this_interval_start - input_iterator);
        }
    } /* For every synapse */
  }

  static void iterate(const google::protobuf::RepeatedPtrField<Interval_type>
                          &arg_synapse_interval,
                      std::function<void(Interval_type)> do_for_each_synapse,
                      std::function<void(std::int32_t)> do_for_each_index,
                      std::uint32_t interval_start = 0,
                      std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter) {
      do_for_each_synapse(arg_synapse_interval[syn_iter]);
      std::uint32_t this_interval_size =
          arg_synapse_interval[syn_iter].interval_size();
      std::uint32_t this_interval_start =
          arg_synapse_interval[syn_iter].starts();
      if (!is_index_input(this_interval_start)) {
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator) {
          do_for_each_index(this_interval_start + input_iterator);
        }
      } else { /* current @starts. element is from the input, iterate in a
                  negative way */
        for (std::int32_t input_iterator = 0;
             static_cast<std::uint32_t>(input_iterator) < this_interval_size;
             ++input_iterator) {
          do_for_each_index(this_interval_start - input_iterator);
        }
      }
    } /* For every synapse */
  }

  static void
  skim_terminatable(const google::protobuf::RepeatedPtrField<Interval_type>
                        &arg_synapse_interval,
                    std::function<bool(Interval_type)> do_for_each_synapse,
                    std::uint32_t interval_start = 0,
                    std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter)
      if (!do_for_each_synapse(arg_synapse_interval[syn_iter]))
        return;
  }

  static void
  iterate_terminatable(const google::protobuf::RepeatedPtrField<Interval_type>
                           &arg_synapse_interval,
                       std::function<bool(std::int32_t)> do_for_each_index,
                       std::uint32_t interval_start = 0,
                       std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter) {
      std::uint32_t this_interval_size =
          arg_synapse_interval[syn_iter].interval_size();
      std::uint32_t this_interval_start =
          arg_synapse_interval[syn_iter].starts();
      if (!is_index_input(this_interval_start)) {
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator)
          if (!do_for_each_index(this_interval_start + input_iterator))
            return;
      } else { /* current @starts. element is from the input, iterate in a
                  negative way */
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator)
          if (!do_for_each_index(this_interval_start - input_iterator))
            return;
      }
    } /* For every synapse */
  }

  static void
  iterate_terminatable(const google::protobuf::RepeatedPtrField<Interval_type>
                           &arg_synapse_interval,
                       std::function<bool(Interval_type)> do_for_each_synapse,
                       std::function<bool(std::int32_t)> do_for_each_index,
                       std::uint32_t interval_start = 0,
                       std::uint32_t interval_size_ = 0) {
    std::uint32_t interval_size = get_number_of_synapses_to_iterate(
        arg_synapse_interval, interval_start, interval_size_);
    for (std::uint32_t syn_iter = interval_start;
         syn_iter < (interval_start + interval_size); ++syn_iter) {
      if (!do_for_each_synapse(arg_synapse_interval[syn_iter]))
        return;
      std::uint32_t this_interval_size =
          arg_synapse_interval[syn_iter].interval_size();
      std::uint32_t this_interval_start =
          arg_synapse_interval[syn_iter].starts();
      if (!is_index_input(this_interval_start)) {
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator)
          if (!do_for_each_index(this_interval_start + input_iterator))
            return;
      } else { /* current @starts. element is from the input, iterate in a
                  negative way */
        for (std::uint32_t input_iterator = 0;
             input_iterator < this_interval_size; ++input_iterator)
          if (!do_for_each_index(this_interval_start - input_iterator))
            return;
      }
    } /* For every synapse */
  }

  /**
   * @brief      Direct access to an indvidual synapse index.
   *
   * @param[in]  index  The index
   *
   * @return     The Synapse index under the @index-th step into the iteration
   */
  std::int32_t operator[](uint32_t index) const {
    RFASSERT(index < size());
    std::int32_t result_index;
    std::uint32_t previous_last_reached_index = 0;
    std::uint32_t iteration_helper = 0;
    std::uint32_t synapse_start = 0;

    if (m_lastReachedIndex <= index) {
      synapse_start = m_lastReachedSynapse;
      iteration_helper = m_lastReachedIndex;
    } else
      m_lastReachedSynapse = 0;

    iterate_terminatable(
        [&](Interval_type /*interval_synapse*/) {
          ++m_lastReachedSynapse;
          m_lastReachedIndex = iteration_helper;
          previous_last_reached_index = m_lastReachedIndex;
          return true;
        },
        [&](std::int32_t synapse_index) {
          if (iteration_helper < index) {
            ++iteration_helper;
            return true;
          } else {
            result_index = synapse_index;
            return false;
          }
        },
        synapse_start);
    RFASSERT(iteration_helper == index);
    --m_lastReachedSynapse;
    m_lastReachedIndex = previous_last_reached_index;
    return result_index;
  }

  /**
   * @brief     Provides the index value the given interval would start at if
   * the interval sizes were considered expanded(not compressed into (start,
   * size) pairs, rather each index separately)
   *
   * @param     interval_index    The index of the interval to query
   *
   * @return  the index to the first element inside the interval given by
   * @interval_index, when operator[] is used
   */
  std::uint32_t interval_starts_at(std::uint32_t interval_index) const {
    RFASSERT(static_cast<std::int32_t>(interval_index) <
             m_synapseInterval.size());
    std::uint32_t index = 0;
    for (std::uint32_t syn_iter = 0u; syn_iter < interval_index; ++syn_iter) {
      index += m_synapseInterval[syn_iter].interval_size();
    }
    return index;
  }

  std::uint32_t interval_size_of(std::uint32_t nth_element) {
    RFASSERT(0u < size());
    std::uint32_t result_size;
    std::uint32_t previous_last_reached_index = 0;
    std::uint32_t iteration_helper = 0;
    std::uint32_t synapse_start = 0;

    if (m_lastReachedIndex <= nth_element) {
      synapse_start = m_lastReachedSynapse;
      iteration_helper = m_lastReachedIndex;
    } else
      m_lastReachedSynapse = 0;

    iterate_terminatable(
        [&](InputSynapseInterval interval_synapse) {
          ++m_lastReachedSynapse;
          m_lastReachedIndex = iteration_helper;
          previous_last_reached_index = m_lastReachedIndex;
          result_size = interval_synapse.interval_size();
          return true;
        },
        [&](std::int32_t /*synapse_index*/) {
          if (iteration_helper < nth_element) {
            ++iteration_helper;
            return true;
          } else
            return false; /* queired nth_element reached, no need to continue */
        },
        synapse_start);
    RFASSERT(iteration_helper == nth_element);
    --m_lastReachedSynapse;
    m_lastReachedIndex = previous_last_reached_index;
    return result_size;
  }

  template <typename InputSynapseInterval>
  std::uint32_t reach_past_loops(std::uint32_t nth_element) const {
    RFASSERT(0u < size());
    std::uint32_t result_reach;
    std::uint32_t previous_last_reached_index = 0;
    std::uint32_t iteration_helper = 0;
    std::uint32_t synapse_start = 0;

    if (m_lastReachedIndex <= nth_element) {
      synapse_start = m_lastReachedSynapse;
      iteration_helper = m_lastReachedIndex;
    } else
      m_lastReachedSynapse = 0;

    std::int32_t synapse_reach;
    iterate_terminatable(
        [&](InputSynapseInterval interval_synapse) {
          ++m_lastReachedSynapse;
          m_lastReachedIndex = iteration_helper;
          previous_last_reached_index = m_lastReachedIndex;
          synapse_reach = interval_synapse.reach_past_loops();
          return true;
        },
        [&](std::int32_t /*synapse_index*/) {
          if (iteration_helper < nth_element) {
            ++iteration_helper;
            return true;
          } else {
            result_reach = synapse_reach;
            return false;
          }
        },
        synapse_start);
    RFASSERT(iteration_helper == nth_element);
    --m_lastReachedSynapse;
    m_lastReachedIndex = previous_last_reached_index;
    return result_reach;
  }

  /**
   * @brief      Equality operator with another synapse: the two will match when
   * all of their synapses are the same
   *
   * @param[in]  index  The other iterator to compare to
   *
   * @return     true if the two SynapseIterators match
   */
  bool operator==(const SynapseIterator<Interval_type> &other) const {
    std::uint32_t synapse_index = 0u;
    std::uint32_t match = true;
    iterate_terminatable([other, &synapse_index, &match](std::int32_t index) {
      if (index != other[synapse_index]) {
        match = false;
        return false;
      }
      ++synapse_index;
      return true;
    });
    return match;
  }

  /**
   * @brief      Non-equality operator with another synapse
   *
   * @param[in]  index  The other iterator to compare to
   *
   * @return     true if the two SynapseIterators matchn't
   */
  bool operator!=(const SynapseIterator<Interval_type> &other) const {
    return !(*this == other);
  }

  /**
   * @brief      Gives back a copy of the synapse under the given index:
   *             In a synapse with multiple intervals, the index refers to
   *             the number of contained indices, not the number of intervals.
   *
   * @param[in]  index  The index
   *
   * @return     The interval synapse
   */
  Interval_type synapse_under(std::int32_t index) {
    if (0 == size())
      throw "Empty synapse iterator reached for query!";
    std::int32_t iteration_helper = 0;
    Interval_type result;

    iterate_terminatable(
        [&](Interval_type interval_synapse) {
          result = interval_synapse;
          return true;
        },
        [&](std::int32_t /*synapse_index*/) {
          if (iteration_helper < index) {
            ++iteration_helper;
            return true; /* Found the synapse we have been looking for */
          } else
            return false; /* Continue searching.. */
        });
    if (iteration_helper != index)
      throw "Index Out of bounds with Synapse Iterator!";
    return result;
  }

  /**
   * @brief      Returns the overall number of inputs
   *
   * @return     Returns the overall number of inputs
   */
  std::uint32_t size() const {
    std::uint32_t number_of_inputs = 0;
    skim([&](Interval_type interval) {
      number_of_inputs += interval.interval_size();
    });
    return number_of_inputs;
  }

  /**
   * @brief      Refresh the cache variable for @cached_size
   *
   */
  void refresh_cached_size() {
    m_cachedSizeVar = size();
    RFASSERT_LOG("Refreshing caches size in synapse iterator to: {}",
                 m_cachedSizeVar);
  }

  /**
   * @brief      Return the number of input synapses in the iterator.
   *             !!!IMPORTANT!!! This provides a cahce variable which
   *             needs to be kept up to date with @refresh_cached_size
   *
   * @return     Size of the Repeatedfiled for the  synapse_intervals
   */
  constexpr std::uint32_t cached_size() const { return m_cachedSizeVar; }

  /**
   * @brief      Return the number of input synapses in the iterator.
   *
   * @return     Size of the Repeatedfiled for the  synapse_intervals
   */
  std::uint32_t number_of_synapses() const { return m_synapseInterval.size(); }

  /**
   * @brief      Give back the last element of the synapse
   *
   * @return     the last index of the synapse
   */
  std::int32_t back() const {
    if (0 < m_synapseInterval.size()) {
      std::int32_t last_index =
          m_synapseInterval[m_synapseInterval.size() - 1].starts();
      if (is_index_input(last_index))
        last_index -=
            m_synapseInterval[m_synapseInterval.size() - 1].interval_size() - 1;
      else
        last_index +=
            m_synapseInterval[m_synapseInterval.size() - 1].interval_size() - 1;
      return last_index;
    } else
      throw std::runtime_error("Last index requested from empty synapse!");
  }

  /**
   * @brief      Add back the last stored synapse interval
   *
   * @return     Synapse interval defined by template function
   */
  Interval_type last_synapse() const {
    if (0 < m_synapseInterval.size()) {
      return m_synapseInterval[m_synapseInterval.size() - 1];
    } else
      throw std::runtime_error("Last item requested from empty synapse!");
  }

  /**
   * @brief      Determines whether the specified index is index taken from the
   * inputs, rather than internally.
   *
   * @param[in]  index  The index
   *
   * @return     True if the specified index is index input, False otherwise.
   */
  static constexpr bool is_index_input(std::int32_t index) {
    return (0 > index);
  }

  /**
   * @brief      Determines whether the specified input interval is pointing to
   * the inputs, rather than internal Neuron data.
   *
   * @param[in]  interval   A const reference to the interval to examine
   *
   * @return     True if the specified interval is pointing to network input,
   * False otherwise.
   */
  static constexpr bool is_synapse_input(const Interval_type &interval) {
    return (is_index_input(interval.starts()));
  }

  /**
   * @brief      Converts synapse an array index[0..n] to an index usable inside
   * synapses where index values from two different sources are merged together
   * into one number and the external part is represented by index values
   * [-n+1..-1]
   *
   * @param[in]  index    the index[0..n] to be converted to external array
   * range
   *
   * @return     index[-n+1..-1] converted to external range
   */
  static constexpr std::int32_t
  external_index_from_array_index(std::uint32_t index) {
    return (static_cast<std::int32_t>(index) * (-1) - 1);
  }

  /**
   * @brief      Converts index values from array index to external index
   *             where index values from two different sources are merged
   * together into one number and the external part is represented by index
   * values [-n+1..-1]
   *
   * @param[in]  index  index[-n+1..-1] converted to external range to be
   * converted
   *
   * @return     the index[0..n] converted to be in internal array range
   */
  static constexpr std::uint32_t
  array_index_from_external_index(std::int32_t index) {
    if (0 > index)
      return (static_cast<std::uint32_t>(index) * (-1) - 1);
    else
      throw std::runtime_error("Synapse index is not negative, as it should "
                               "be, when queried for input index! ");
  }

private:
  const google::protobuf::RepeatedPtrField<Interval_type> &m_synapseInterval;
  mutable std::uint32_t m_lastReachedSynapse = 0u;
  mutable std::uint32_t m_lastReachedIndex = 0u;
  static std::uint32_t m_intervalSize; /* temporary variable */
  std::uint32_t m_cachedSizeVar;

  /**
   * @brief      Gets the number of synapses to iterate based on the provided
   * start and size values. Checks wether the arguments related validity, and
   * returns the number of synapses to iterate over. It sets the mutable helper
   * @interval_size to the correct iteration value
   *
   * @param      arg_synapse_interval  The argument synapse interval
   * @param[in]  interval_start        The interval start
   * @param[in]  interval_size_        The interval size
   *
   */
  static constexpr std::uint32_t get_number_of_synapses_to_iterate(
      const google::protobuf::RepeatedPtrField<Interval_type>
          &arg_synapse_interval,
      std::uint32_t interval_start, std::uint32_t interval_size_) {
    if (0 == arg_synapse_interval.size())
      return 0;
    else if ((0 == interval_size_) &&
             (arg_synapse_interval.size() >
              static_cast<std::int32_t>(interval_start)))
      return (arg_synapse_interval.size() - interval_start);
    else if (0 == interval_size_)
      throw std::runtime_error("Incorrect synapse range start!");
    else
      return interval_size_;
  }
};

} /* namespace rafko_net */

#endif /* SYNAPSE_ITERATOR_H */
