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
class RAFKO_FULL_EXPORT RafkoDatasetWrapper{
public:
  explicit RafkoDatasetWrapper(const rafko_gym::DataSet& samples_)
  : sequence_size(std::max(1u,samples_.sequence_size()))
  , input_samples(samples_.inputs_size() / samples_.input_size())
  , label_samples(samples_.labels_size() / samples_.feature_size())
  , prefill_sequences(static_cast<uint32>((samples_.inputs_size() - samples_.labels_size()) / (samples_.labels_size() / sequence_size)))
  {
    assert(0 == (label_samples.size()%sequence_size));
    fill(samples_);
  }

  RafkoDatasetWrapper(std::vector<std::vector<sdouble32>>&& input_samples_, std::vector<std::vector<sdouble32>>&& label_samples_, uint32 sequence_size_ = 1u)
  : sequence_size(std::max(1u,sequence_size_))
  , input_samples(std::move(input_samples_))
  , label_samples(std::move(label_samples_))
  , prefill_sequences(static_cast<uint32>((input_samples.size() - label_samples.size()) / (label_samples.size() / sequence_size)))
  { assert(0 == (label_samples.size()%sequence_size)); }

  /**
   * @brief      Gets an input sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The input sample.
   */
  const std::vector<sdouble32>& get_input_sample(uint32 raw_input_index) const{
    assert(input_samples.size() > raw_input_index);
    return input_samples[raw_input_index];
  }

  /**
   * @brief      Gets an input sample from the set
   *
   * @return     A const reference of the input sample.
   */
  const std::vector<std::vector<sdouble32>>& get_input_samples() const{
    return input_samples;
  }

  /**
   * @brief      Gets a label sample from the set
   *
   * @param[in]  sample_index  The sample index
   *
   * @return     The label sample.
   */
  const std::vector<sdouble32>& get_label_sample(uint32 raw_label_index) const{
    assert(label_samples.size() > raw_label_index);
    return label_samples[raw_label_index];
  }

  /**
   * @brief      Gets a label sample from the set
   *
   * @return     A const reference of the label samples array
   */
  const std::vector<std::vector<sdouble32>>& get_label_samples() const{
    return label_samples;
  }

  /**
   * @brief      Gets the number of values present in the output
   *
   * @return     The feature size.
   */
  uint32 get_feature_size() const{
    return label_samples[0].size();
  }

  /**
   * @brief      Gets the number of raw input arrays stored in the pbject
   *
   * @return     The number of input samples.
   */
  uint32 get_number_of_input_samples() const{
    return input_samples.size();
  }

  /**
   * @brief      The number of raw label arrays stored in the object
   *
   * @return     The number of labels.
   */
  uint32 get_number_of_label_samples() const{
    return label_samples.size();
  }

  /**
   * @brief      Gets the number of sequences stored in the object. One sequence contains
   *             a number of input and label sample arrays. There might be more input arrays,
   *             than label arrays in one sequences. The difference is given by @get_prefill_inputs_number
   *
   * @return     The number of sequences.
   */
  uint32 get_number_of_sequences() const{
    return (get_number_of_label_samples() / sequence_size);
  }

  /**
   * @brief      Gets the size of one sequence
   *
   * @return     Number of consecutive datapoints that count as one sample.
   */
  uint32 get_sequence_size() const{
    return sequence_size;
  }

  /**
   * @brief      Gets the number of inputs to be used as initializing the network during a training run
   *
   * @return     The number of inputs to be used for network initialization during training
   */
  uint32 get_prefill_inputs_number() const{
    return prefill_sequences;
  }

private:
  const uint32 sequence_size;
  std::vector<std::vector<sdouble32>> input_samples;
  std::vector<std::vector<sdouble32>> label_samples;
  const uint32 prefill_sequences; /* Number of input sequences used only to create an initial state for the Neural network */

  /**
   * @brief      Converting the @rafko_gym::DataSet message to vectors
   *
   * @param      samples  The data set to parse
   */
  void fill(const rafko_gym::DataSet& samples);
};

} /* namespace rafko_gym */

#endif /* RAFKO_DATASET_WRAP_H */
