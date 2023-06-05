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
#include "rafko_gym/models/rafko_cost.hpp"
#include "cost_function_binary_cross_entropy.hpp"
#include "cost_function_cross_entropy.hpp"
#include "cost_function_mse.hpp"
#include "cost_function_squared_error.hpp"

#include <math.h>
#include <mutex>
#include <functional>


#if(RAFKO_USES_OPENCL)
#include "rafko_utilities/services/rafko_string_utils.hpp"
#include "rafko_gym/services/function_factory.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"


namespace rafko_gym{

rafko_utilities::DataPool<double> RafkoCost::m_commonDatapool(1,1);

double RafkoCost::set_feature_for_label(
  const rafko_gym::RafkoDataSet& environment, std::uint32_t sample_index,
  const std::vector<double>& neuron_data
) const{
  RFASSERT(environment.get_number_of_label_samples() > sample_index);
  return m_costFunction->get_feature_error(
    environment.get_label_sample(sample_index), neuron_data,
    environment.get_number_of_label_samples()
  );
}

double RafkoCost::set_features_for_labels(
  const rafko_gym::RafkoDataSet& environment, const std::vector<std::vector<double>>& neuron_data,
  std::uint32_t neuron_buffer_start_index, std::uint32_t raw_start_index, std::uint32_t labels_to_evaluate
) const{
  RFASSERT((raw_start_index + labels_to_evaluate) <= environment.get_number_of_label_samples());

  std::vector<double>& error_labels = m_commonDatapool.reserve_buffer(labels_to_evaluate);
  /*!Note: No need to fill the reserved buffer with initial values because every element of it will be overwritten */
  m_costFunction->get_feature_errors(
    environment.get_label_samples(), neuron_data, error_labels,
    raw_start_index, 0/* error start */, labels_to_evaluate, neuron_buffer_start_index,
    environment.get_number_of_label_samples()
  );

  double error_sum = (0.0);
  std::mutex sum_mutex;
  m_errorCalculationThreads.start_and_block( std::bind( &RafkoCost::accumulate_error_sum, this,
    std::ref(error_labels), std::ref(error_sum), labels_to_evaluate,
    std::ref(sum_mutex), std::placeholders::_1
  ) );
  m_commonDatapool.release_buffer(error_labels);
  return error_sum;
}

double RafkoCost::set_features_for_sequences(
  const rafko_gym::RafkoDataSet& environment, const std::vector<std::vector<double>>& neuron_data,
  std::uint32_t neuron_buffer_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
  std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation
) const{
  std::vector<double>& resulting_errors = m_commonDatapool.reserve_buffer(sequences_to_evaluate * environment.get_sequence_size());
  double error_sum = set_features_for_sequences(
    environment, neuron_data,
    neuron_buffer_index, sequence_start_index, sequences_to_evaluate,
    start_index_in_sequence, sequence_truncation, resulting_errors
  );
  m_commonDatapool.release_buffer(resulting_errors);
  return error_sum;
}

double RafkoCost::set_features_for_sequences(
  const rafko_gym::RafkoDataSet& environment, const std::vector<std::vector<double>>& neuron_data,
  std::uint32_t neuron_buffer_start_index, std::uint32_t sequence_start_index, std::uint32_t sequences_to_evaluate,
  std::uint32_t start_index_in_sequence, std::uint32_t sequence_truncation, std::vector<double>& tmp_data
) const{
  RFASSERT(environment.get_number_of_sequences() >= (sequence_start_index + sequences_to_evaluate));
  RFASSERT(environment.get_sequence_size() >= (start_index_in_sequence + sequence_truncation));

  std::uint32_t raw_start_index = sequence_start_index * environment.get_sequence_size();
  std::uint32_t labels_to_evaluate = sequences_to_evaluate * environment.get_sequence_size();
  std::vector<double>& error_labels = m_commonDatapool.reserve_buffer(labels_to_evaluate);

  tmp_data.resize(labels_to_evaluate);
  std::fill(tmp_data.begin(), tmp_data.end(), (0.0));
  std::fill(error_labels.begin(), error_labels.end(), (0.0));

  m_costFunction->get_feature_errors(
    environment.get_label_samples(), neuron_data, tmp_data,
    raw_start_index, 0/* error_start */, labels_to_evaluate, neuron_buffer_start_index,
    environment.get_number_of_label_samples()
  );

  std::uint32_t copy_index_base = 0;
  for(std::uint32_t sequence_iterator = 0; sequence_iterator < sequences_to_evaluate; ++sequence_iterator){
    std::copy(
      tmp_data.begin() + copy_index_base + start_index_in_sequence,
      tmp_data.begin() + copy_index_base + start_index_in_sequence + sequence_truncation,
      error_labels.begin() + copy_index_base + start_index_in_sequence
    );
    copy_index_base += environment.get_sequence_size();
  }

  double error_sum = (0.0);
  std::mutex sum_mutex;
  m_errorCalculationThreads.start_and_block( std::bind( &RafkoCost::accumulate_error_sum, this,
    std::ref(error_labels), std::ref(error_sum), labels_to_evaluate,
    std::ref(sum_mutex), std::placeholders::_1
  ) );
  m_commonDatapool.release_buffer(error_labels);
  return error_sum;
}

void RafkoCost::accumulate_error_sum(std::vector<double>& source, double& target, std::uint32_t length, std::mutex& error_mutex, std::uint32_t thread_index)  const{
  double local_error = (0.0);
  std::uint32_t errors_to_sum_in_one_thread = (length / m_settings.get_sqrt_of_solve_threads()) + 1;
  std::uint32_t error_start = errors_to_sum_in_one_thread * thread_index;
  errors_to_sum_in_one_thread = std::min(errors_to_sum_in_one_thread, (length - error_start));
  errors_to_sum_in_one_thread = std::min(errors_to_sum_in_one_thread, static_cast<std::uint32_t>(length - error_start));

  for(std::uint32_t sample_index = error_start; sample_index < (error_start + errors_to_sum_in_one_thread) ; ++sample_index)
    local_error += source[sample_index];

  std::lock_guard<std::mutex> my_lock(error_mutex);
  target += local_error;
}

std::string RafkoCost::generic_derivative_kernel_source(
  std::string label_value, std::string feature_value, std::string feature_d, std::string sample_number,
  std::string target, std::string behavior_index
){
  std::string kernel_source = R"(
    switch(==behavior_index==){
      case cost_function_squared_error:{==target== = ==cost_function_squared_error==; }break;
      case cost_function_mse:{==target== = ==cost_function_mse==; }break;
      case cost_function_cross_entropy:{==target== = ==cost_function_cross_entropy==; }break;
      case cost_function_binary_cross_entropy:{==target== = ==cost_function_binary_cross_entropy==; }break;
    }
  )";
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==target=="), target);
  kernel_source = rafko_utilities::replace_all_in_string(kernel_source, std::regex("==behavior_index=="), behavior_index);
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==cost_function_squared_error=="),
    CostFunctionSquaredError::derivative_kernel_source(label_value, feature_value, feature_d)
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==cost_function_mse=="), 
    CostFunctionMSE::derivative_kernel_source(label_value, feature_value, feature_d, sample_number)
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==cost_function_cross_entropy=="), 
    CostFunctionCrossEntropy::derivative_kernel_source(label_value, feature_value, feature_d, sample_number)
  );
  kernel_source = rafko_utilities::replace_all_in_string(
    kernel_source, std::regex("==cost_function_binary_cross_entropy=="), 
    CostFunctionBinaryCrossEntropy::get_derivative_kernel_source(label_value, feature_value, feature_d, sample_number)
  );
  return kernel_source;
}

} /* namespace rafko_gym */
