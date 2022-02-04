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
  , prefill_sequences( static_cast<uint32>((input_samples.size() - label_samples.size())) / (label_samples.size() / sequence_size) )
  {
    assert(0 == (label_samples.size()%sequence_size));
    assert(0 < samples_.input_size());
    assert(0 < samples_.feature_size());
    assert(0 < samples_.sequence_size());
    assert(0 < samples_.inputs_size());
    assert(0 < samples_.labels_size());
    fill(samples_);
  }

  RafkoDatasetWrapper(std::vector<std::vector<sdouble32>>&& input_samples_, std::vector<std::vector<sdouble32>>&& label_samples_, uint32 sequence_size_ = 1u)
  : sequence_size(std::max(1u,sequence_size_))
  , input_samples(std::move(input_samples_))
  , label_samples(std::move(label_samples_))
  , prefill_sequences(static_cast<uint32>((input_samples.size() - label_samples.size()) / (label_samples.size() / sequence_size)))
  {
    assert(0 == (label_samples.size()%sequence_size));
    assert(0 < input_samples.size());
    assert(input_samples.size() == label_samples.size());
  }

  const std::vector<sdouble32>& get_input_sample(uint32 raw_input_index) const{
    assert(input_samples.size() > raw_input_index);
    return input_samples[raw_input_index];
  }

  const std::vector<std::vector<sdouble32>>& get_input_samples() const{
    return input_samples;
  }

  const std::vector<sdouble32>& get_label_sample(uint32 raw_label_index) const{
    assert(label_samples.size() > raw_label_index);
    return label_samples[raw_label_index];
  }

  const std::vector<std::vector<sdouble32>>& get_label_samples() const{
    return label_samples;
  }

  uint32 get_feature_size() const{
    return label_samples[0].size();
  }

  uint32 get_input_size() const{
    return input_samples[0].size();
  }

  uint32 get_number_of_input_samples() const{
    return input_samples.size();
  }

  uint32 get_number_of_label_samples() const{
    return label_samples.size();
  }

  uint32 get_number_of_sequences() const{
    return (get_number_of_label_samples() / sequence_size);
  }

  uint32 get_sequence_size() const{
    return sequence_size;
  }

  uint32 get_prefill_inputs_number() const{
    return prefill_sequences;
  }

  /*!Note: There's no state to talk about with this specialization */
  void push_state(){ }
  void pop_state(){ }

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