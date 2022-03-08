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

#ifndef RAFKO_DATASET_WRAP_H
#define RAFKO_DATASET_WRAP_H

#include "rafko_global.h"

#include <vector>
#include <math.h>

#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_mainframe/services/rafko_assertion_logger.h"

namespace rafko_gym{

/**
 * @brief      A wrapper class to store @DataSets in a friendly format.
 *             It is possible to have more input samples, than label samples; In those cases
 *             the extra inputs are to be used to initialize the network before training.
 *             The data set consists of labels and inputs. Not every label has an input assigned to it,
 *             as there might be some additional inputs used to "prefill" a network, setting it up to be
 *             evaluated by the labels. It helps setting up an initial state for the training.
 *             The Dataset is built up of multiple sequences, each input and label in the sequence
 *             is of the same size and dimension. Each input and label can be of any size, albeit they must have
 *             the same size or every sample.
 *             ================================================
 *             Example of the structure:
 *             Dataset ( prefill 2, sequence size 6:
 *             - Sequence (sample) 1:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 2:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 3:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             - Sequence (sample) 4:
 *             - Inputs: [][][][][][]
 *             - Labels:     [][][][]
 *             ================================================
 *             Despite the above structure, for eligibility of paralellism, the inputs and labels are in a separate,
 *             contigous array.
 */
class RAFKO_FULL_EXPORT RafkoDatasetWrapper : public RafkoEnvironment{
public:
  explicit RafkoDatasetWrapper(const rafko_gym::DataSet& samples_)
  : sequence_size(std::max(1u,samples_.sequence_size()))
  , input_samples(samples_.inputs_size() / samples_.input_size())
  , label_samples(samples_.labels_size() / samples_.feature_size())
  , prefill_sequences( static_cast<std::uint32_t>((input_samples.size() - label_samples.size())) / (label_samples.size() / sequence_size) )
  {
    RFASSERT(0 == (label_samples.size()%sequence_size));
    RFASSERT(0 < samples_.input_size());
    RFASSERT(0 < samples_.feature_size());
    RFASSERT(0 < samples_.sequence_size());
    RFASSERT(0 < samples_.inputs_size());
    RFASSERT(0 < samples_.labels_size());
    fill(samples_);
  }

  RafkoDatasetWrapper(std::vector<std::vector<double>>&& input_samples_, std::vector<std::vector<double>>&& label_samples_, std::uint32_t sequence_size_ = 1u)
  : sequence_size(std::max(1u,sequence_size_))
  , input_samples(std::move(input_samples_))
  , label_samples(std::move(label_samples_))
  , prefill_sequences(static_cast<std::uint32_t>((input_samples.size() - label_samples.size()) / (label_samples.size() / sequence_size)))
  {
    RFASSERT(0 == (label_samples.size()%sequence_size));
    RFASSERT(0 < input_samples.size());
    RFASSERT(input_samples.size() == label_samples.size());
  }

  const std::vector<double>& get_input_sample(std::uint32_t raw_input_index) const{
    RFASSERT(input_samples.size() > raw_input_index);
    return input_samples[raw_input_index];
  }

  constexpr const std::vector<std::vector<double>>& get_input_samples() const{
    return input_samples;
  }

  const std::vector<double>& get_label_sample(std::uint32_t raw_label_index) const{
    RFASSERT(label_samples.size() > raw_label_index);
    return label_samples[raw_label_index];
  }

  const std::vector<std::vector<double>>& get_label_samples() const{
    return label_samples;
  }

  std::uint32_t get_feature_size() const{
    return label_samples[0].size();
  }

  std::uint32_t get_input_size() const{
    return input_samples[0].size();
  }

  std::uint32_t get_number_of_input_samples() const{
    return input_samples.size();
  }

  std::uint32_t get_number_of_label_samples() const{
    return label_samples.size();
  }

  std::uint32_t get_number_of_sequences() const{
    return (get_number_of_label_samples() / sequence_size);
  }

  constexpr std::uint32_t get_sequence_size() const{
    return sequence_size;
  }

  constexpr std::uint32_t get_prefill_inputs_number() const{
    return prefill_sequences;
  }

  /*!Note: There's no state to talk about with this specialization */
  constexpr void push_state(){ }
  constexpr void pop_state(){ }

private:
  const std::uint32_t sequence_size;
  std::vector<std::vector<double>> input_samples;
  std::vector<std::vector<double>> label_samples;
  const std::uint32_t prefill_sequences; /* Number of input sequences used only to create an initial state for the Neural network */

  /**
   * @brief      Converting the @rafko_gym::DataSet message to vectors
   *
   * @param      samples  The data set to parse
   */
  void fill(const rafko_gym::DataSet& samples);
};

} /* namespace rafko_gym */

#endif /* RAFKO_DATASET_WRAP_H */
