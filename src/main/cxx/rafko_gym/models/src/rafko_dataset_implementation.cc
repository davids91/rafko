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
#include "rafko_gym/models/rafko_dataset_implementation.hpp"

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym{

void RafkoDatasetImplementation::fill(
  const DataSetPackage& samples, std::vector<FeatureVector>& input_samples, std::vector<FeatureVector>& label_samples
){
  std::uint32_t feature_start_index = 0;
  std::uint32_t input_start_index = 0;
  input_samples.resize(samples.inputs_size() / samples.input_size());
  label_samples.resize(samples.labels_size() / samples.feature_size());

  /*!Note: One cycle can be used for both, because there will always be at least as many inputs as labels */
  for(std::uint32_t raw_sample_iterator = 0; raw_sample_iterator < input_samples.size(); ++ raw_sample_iterator){
    input_samples[raw_sample_iterator] = std::vector<double>(samples.input_size());
    for(std::uint32_t input_iterator = 0; input_iterator < samples.input_size(); ++input_iterator)
      input_samples[raw_sample_iterator][input_iterator] = samples.inputs(input_start_index + input_iterator);
    input_start_index += samples.input_size();
    if(raw_sample_iterator < label_samples.size()){
      label_samples[raw_sample_iterator] = std::vector<double>(samples.feature_size());
      for(std::uint32_t feature_iterator = 0; feature_iterator < samples.feature_size(); ++feature_iterator)
        label_samples[raw_sample_iterator][feature_iterator] = samples.labels(feature_start_index + feature_iterator);
      feature_start_index += samples.feature_size();
    }
  }
}

DataSetPackage RafkoDatasetImplementation::generate_from(
  const std::vector<FeatureVector>& input_samples, const std::vector<FeatureVector>& label_samples,
  std::uint32_t sequence_size, std::uint32_t possible_sequence_count
){
  RFASSERT(0 < input_samples.size());
  RFASSERT(0 < label_samples.size());
  RFASSERT(0 == (label_samples.size()%sequence_size));
  RFASSERT(
    (0 == possible_sequence_count) || ((label_samples.size() / sequence_size) <= possible_sequence_count)
  );

  DataSetPackage result;
  result.set_input_size(input_samples.size());
  result.set_feature_size(label_samples.size());
  result.set_sequence_size(sequence_size);
  result.set_possible_sequence_count(possible_sequence_count);
  for(std::uint32_t sample_index = 0; sample_index < input_samples.size(); ++sample_index){
    result.mutable_inputs()->Add(input_samples[sample_index].begin(),input_samples[sample_index].end());
    result.mutable_labels()->Add(label_samples[sample_index].begin(),label_samples[sample_index].end());
  }
}


} /* namespace rafko_gym */
