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

#ifndef RAFKO_DATASET_WRAPPER_H
#define RAFKO_DATASET_WRAPPER_H

#include "rafko_global.hpp"

#include <vector>
#include <math.h>

#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym{

/**
 * @brief      A wrapper class to store @DataSetPackages in a friendly format.
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
class RAFKO_EXPORT RafkoDatasetImplementation : public RafkoDataSet{
public:
  explicit RafkoDatasetImplementation(const rafko_gym::DataSetPackage& samples)
  : m_sequenceSize(std::max(1u,samples.sequence_size()))
  , m_inputSamples(samples.inputs_size() / samples.input_size())
  , m_labelSamples(samples.labels_size() / samples.feature_size())
  , m_prefillSequences( static_cast<std::uint32_t>((m_inputSamples.size() - m_labelSamples.size())) / (m_labelSamples.size() / m_sequenceSize) )
  {
    RFASSERT(0 == (m_labelSamples.size()%m_sequenceSize));
    RFASSERT(0 < samples.input_size());
    RFASSERT(0 < samples.feature_size());
    RFASSERT(0 < samples.sequence_size());
    RFASSERT(0 < samples.inputs_size());
    RFASSERT(0 < samples.labels_size());
    fill(samples);
  }

  RafkoDatasetImplementation(std::vector<std::vector<double>>&& input_samples, std::vector<std::vector<double>>&& label_samples, std::uint32_t sequence_size = 1u)
  : m_sequenceSize(std::max(1u,sequence_size))
  , m_inputSamples(std::move(input_samples))
  , m_labelSamples(std::move(label_samples))
  , m_prefillSequences(static_cast<std::uint32_t>((m_inputSamples.size() - m_labelSamples.size()) / (m_labelSamples.size() / m_sequenceSize)))
  {
    RFASSERT(0 == (m_labelSamples.size()%m_sequenceSize));
    RFASSERT(0 < m_inputSamples.size());
    RFASSERT(m_inputSamples.size() == m_labelSamples.size());
  }

  const std::vector<double>& get_input_sample(std::uint32_t raw_input_index) const override{
    RFASSERT_LOG("Input sample {} / {}", raw_input_index, m_inputSamples.size());
    RFASSERT(m_inputSamples.size() > raw_input_index);
    return m_inputSamples[raw_input_index];
  }

  constexpr const std::vector<std::vector<double>>& get_input_samples() const override{
    return m_inputSamples;
  }

  const std::vector<double>& get_label_sample(std::uint32_t raw_label_index) const override{
    RFASSERT_LOG("label_sample sample {} / {}", raw_label_index, m_labelSamples.size());
    RFASSERT(m_labelSamples.size() > raw_label_index);
    return m_labelSamples[raw_label_index];
  }

  const std::vector<std::vector<double>>& get_label_samples() const override{
    return m_labelSamples;
  }

  std::uint32_t get_feature_size() const override{
    return m_labelSamples[0].size();
  }

  std::uint32_t get_input_size() const override{
    return m_inputSamples[0].size();
  }

  std::uint32_t get_number_of_input_samples() const override{
    return m_inputSamples.size();
  }

  std::uint32_t get_number_of_label_samples() const override{
    return m_labelSamples.size();
  }

  std::uint32_t get_number_of_sequences() const override{
    return (get_number_of_label_samples() / m_sequenceSize);
  }

  constexpr std::uint32_t get_sequence_size() const override{
    return m_sequenceSize;
  }

  constexpr std::uint32_t get_prefill_inputs_number() const override{
    return m_prefillSequences;
  }

private:
  const std::uint32_t m_sequenceSize;
  std::vector<std::vector<double>> m_inputSamples;
  std::vector<std::vector<double>> m_labelSamples;
  const std::uint32_t m_prefillSequences; /* Number of input sequences used only to create an initial state for the Neural network */

  /**
   * @brief      Converting the @rafko_gym::DataSetPackage message to vectors
   *
   * @param      samples  The data set to parse
   */
  void fill(const rafko_gym::DataSetPackage& samples);
};

} /* namespace rafko_gym */

#endif /* RAFKO_DATASET_WRAPPER_H */
