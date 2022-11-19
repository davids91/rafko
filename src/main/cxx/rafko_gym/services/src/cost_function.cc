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
#include "rafko_gym/services/cost_function.hpp"

#include <math.h>
#include <utility>

#if(RAFKO_USES_OPENCL)
#include "rafko_utilities/models/rafko_gpu_kernel_library.hpp"
#endif/*(RAFKO_USES_OPENCL)*/
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_gym {

void CostFunction::get_feature_errors(
  const std::vector<std::vector<double>>& labels, const std::vector<std::vector<double>>& neuron_data, std::vector<double>& errors_for_labels,
  std::uint32_t label_start, std::uint32_t error_start, std::uint32_t labels_to_evaluate, std::uint32_t neuron_start, std::uint32_t sample_number
) const{
  if((label_start + labels_to_evaluate) > labels.size())
    throw std::runtime_error("Label index out of bounds with Neuron data!");

  if((neuron_data.size() < labels_to_evaluate)||(0 == neuron_data.size()))
    throw std::runtime_error("Can't evaluate more labels, than there is data provided!");

  const std::uint32_t labels_to_do_in_a_thread = 1u + static_cast<std::uint32_t>(labels_to_evaluate/settings.get_sqrt_of_solve_threads());
  m_executionThreads.start_and_block( std::bind( &CostFunction::feature_errors_thread, this,
    std::ref(labels), std::ref(neuron_data), std::ref(errors_for_labels),
    label_start, error_start, neuron_start,
    labels_to_do_in_a_thread, labels_to_evaluate, sample_number, std::placeholders::_1
  ) );
}

void CostFunction::feature_errors_thread(
  const std::vector<std::vector<double>>& labels, const std::vector<std::vector<double>>& neuron_data, std::vector<double>& errors_for_labels,
  std::uint32_t label_start, std::uint32_t error_start, std::uint32_t neuron_data_start_index,
  std::uint32_t labels_to_evaluate_in_one_thread, std::uint32_t labels_evaluating_overall, std::uint32_t sample_number, std::uint32_t thread_index
) const{
  std::uint32_t neuron_data_start_index_in_thread = neuron_data_start_index + (thread_index * labels_to_evaluate_in_one_thread);
  std::uint32_t label_start_in_thread = label_start + (thread_index * labels_to_evaluate_in_one_thread);
  std::uint32_t error_start_in_thread = error_start + (thread_index * labels_to_evaluate_in_one_thread);
  std::int32_t labels_to_evaluate_in_this_thread = std::min( /* Because of the alignment, one thread might include more, .. */
    labels_to_evaluate_in_one_thread,           /* ..than the actual size of the labels/neurons, so labels to evaluate in this thread might.. */
    std::min(                                 /* ..go under 0. No labels are evaluated in this case. */
      static_cast<std::uint32_t>(neuron_data.size() - neuron_data_start_index_in_thread),
      std::min(
        static_cast<std::uint32_t>((label_start + labels_evaluating_overall) - label_start_in_thread),
        static_cast<std::uint32_t>(labels.size() - label_start_in_thread)
      )
    )
  );
  for(std::int32_t label_iterator = 0; label_iterator < labels_to_evaluate_in_this_thread; ++label_iterator){
    errors_for_labels[error_start_in_thread + label_iterator] = get_feature_error(
      labels[label_start_in_thread + label_iterator], neuron_data[neuron_data_start_index_in_thread + label_iterator],
      1, thread_index, sample_number
    );
  }
}

double CostFunction::get_feature_error(
  DataView label, DataView neuron_data,
  std::uint32_t max_threads, std::uint32_t outer_thread_index, std::uint32_t sample_number
) const{
  RFASSERT( label.size() == neuron_data.size() );

  double error_value = 0;
  std::uint32_t feature_start = 0;
  if(max_threads > 1u){
    const std::uint32_t feature_number = 1 + static_cast<std::uint32_t>(label.size()/max_threads);
    for(std::uint32_t thread_index = 0; ((thread_index < max_threads) && (feature_start < label.size())); ++thread_index){
      thread_results[outer_thread_index].push_back(std::async(std::launch::async,
        &CostFunction::summarize_errors, this, std::ref(label), std::ref(neuron_data),
        feature_start, std::min(feature_number, static_cast<std::uint32_t>(label.size() - feature_start))
      ));
      feature_start += feature_number;
    }
    while(0 < thread_results[outer_thread_index].size()){ /* wait for threads */
      error_value += thread_results[outer_thread_index].back().get();
      thread_results[outer_thread_index].pop_back();
    }
  }else{
    error_value = summarize_errors(label, neuron_data, 0, label.size() );
  }

  return error_post_process(error_value, sample_number);
}

double CostFunction::summarize_errors(
  DataView label, DataView neuron_data, std::uint32_t feature_start_index, std::uint32_t number_to_eval
) const{
  double local_error = 0;
  for(std::uint32_t feature_iterator = 0; feature_iterator < number_to_eval; ++feature_iterator)
    local_error += get_cell_error(label[feature_start_index + feature_iterator], neuron_data[feature_start_index + feature_iterator]);
  return local_error;
}

#if(RAFKO_USES_OPENCL)
std::vector<std::string> CostFunction::get_step_names()const  {
  return {"cost_function"};
}

cl::Program::Sources CostFunction::get_step_sources()const {
  std::string source_base = rafko_utilities::atomic_double_add_function + R"(

    void __kernel cost_function(
      __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
      __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      int error_index = get_global_id(0);
      if( (2 == input_sizes_size)&&(input_sizes[0] == input_sizes[1])&&(1 == output_sizes[0]) ){
        const int feature_size = $$feature_size$$;
        const int sample_number = $$sample_number$$;

        if(0 == error_index){ outputs[0] = 0.0; }
        barrier(CLK_GLOBAL_MEM_FENCE);

        double thread_error = 0.0;
        for(int feature_iterator = 0; feature_iterator < feature_size; ++feature_iterator){
          int index = (error_index * feature_size) + feature_iterator;
          thread_error += $$operation_source$$;
        }
        AtomicAdd(&outputs[0], thread_error);

        barrier(CLK_GLOBAL_MEM_FENCE);
        if(0 == error_index)
          outputs[0] = $$post_process_source$$;

      }/*if(IO sizes are correctly set)*/
    }/*kernel*/
  )";

  source_base = std::regex_replace(
    source_base, std::regex("\\$\\$operation_source\\$\\$"),
    get_operation_kernel_source("inputs[input_sizes[0] + index]","inputs[index]")
  );
  source_base = std::regex_replace(
    source_base, std::regex("\\$\\$post_process_source\\$\\$"),
    get_post_process_kernel_source("outputs[0]")
  );
  source_base = std::regex_replace(source_base, std::regex("\\$\\$feature_size\\$\\$"), std::to_string(m_featureSize));
  source_base = std::regex_replace(source_base, std::regex("\\$\\$sample_number\\$\\$"), std::to_string(m_pairsToEvaluate));
  return{source_base};
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_gym */
