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

#include "rafko_mainframe/services/rafko_gpu_context.hpp"

#include <stdexcept>

#include "rafko_mainframe/services/rafko_assertion_logger.hpp"
#include "rafko_mainframe/services/rafko_dummies.hpp"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_protocol/solution.pb.h"

namespace rafko_mainframe {

RafkoGPUContext::RafkoGPUContext(
    cl::Context &&context, cl::Device device,
    rafko_net::RafkoNet &neural_network,
    std::shared_ptr<rafko_mainframe::RafkoSettings> settings,
    std::shared_ptr<rafko_gym::RafkoObjective> objective)
    : RafkoContext(settings), m_network(neural_network),
      m_solverFactory(m_network, m_settings), m_agent(m_solverFactory.build()),
      m_dataSet(std::make_unique<RafkoDummyEnvironment>(
          m_network.input_data_size(), m_network.output_neuron_number())),
      m_objective(objective),
      m_weightUpdater(rafko_gym::UpdaterFactory::build_weight_updater(
          m_network, rafko_gym::weight_updater_default, *m_settings)),
      m_neuronOutputsToEvaluate(/* For every thread, 1 sequence is evaluated..
                                 */
                                (m_settings->get_max_processing_threads() *
                                     m_dataSet->get_sequence_size() +
                                 1u),
                                std::vector<double>(
                                    m_network
                                        .output_neuron_number()) /* ..plus for
                                                                    the label
                                                                    errors one
                                                                    additional
                                                                    vector is
                                                                    needed */
                                ),
      m_executionThreads(m_settings->get_max_processing_threads()),
      m_openclContext(context), m_openclDevice(device),
      m_openclQueue(m_openclContext, m_openclDevice),
      m_solutionPhase(m_openclContext, m_openclDevice, m_openclQueue, m_agent),
      m_errorPhase(m_openclContext, m_openclDevice, m_openclQueue,
                   std::make_shared<RafkoDummyGPUStrategyPhase>(
                       RafkoNBufShape({m_network.output_neuron_number(),
                                       m_network.output_neuron_number()}),
                       RafkoNBufShape({1u}))),
      m_numOutputsInOneSequence(
          std::max({2u, m_network.memory_size(),
                    m_dataSet->get_inputs_in_one_sequence()})),
      m_evalStartInSequence(m_numOutputsInOneSequence -
                            m_dataSet->get_inputs_in_one_sequence()) {
  m_neuronOutputsToEvaluate.back().resize(
      m_dataSet->get_number_of_label_samples());
  upload_weight_table_to_device(); /*!Note: Also sets device_weight_table_size*/
  if (m_objective)
    refresh_objective();
}

void RafkoGPUContext::upload_weight_to_device(std::uint32_t weight_index) {
  const std::vector<std::pair<std::uint32_t, std::uint32_t>>
      &relevant_partial_weights =
          (m_solverFactory.expose_weight_adapter()
               .get_relevant_partial_weight_indices_for(weight_index));
  std::uint32_t weight_table_offset = 0u;
  std::uint32_t partial_index = 0u;
  RFASSERT_LOG("Starting to upload a single weight to device..");
  for (const std::pair<std::uint32_t, std::uint32_t> &index_pair :
       relevant_partial_weights) {
    while (partial_index < std::get<0>(index_pair)) {
      weight_table_offset += m_solverFactory.actual_solution()
                                 ->partial_solutions(partial_index)
                                 .weight_table_size();
      ++partial_index;
    }
    std::uint32_t weight_index_in_partial = std::get<1>(index_pair);
    double weight_value = m_solverFactory.actual_solution()
                              ->partial_solutions(partial_index)
                              .weight_table(weight_index_in_partial);
    RFASSERT_LOG("Weight index in partial[{}]: {}", partial_index,
                 weight_index_in_partial);

    /* Update weight at weight_table_offset + std::get<1>(index_pair) */
    std::uint32_t current_offset =
        (sizeof(double) * (m_agent->get_input_shapes()[0][0] +
                           weight_table_offset + weight_index_in_partial));
    RFASSERT_LOG("buffer byte offset: {} / {}", current_offset,
                 (m_agent->get_input_shapes()[0].get_byte_size<double>()));
    cl_int return_value = m_openclQueue.enqueueWriteBuffer(
        m_solutionPhase.get_input_buffer(), CL_TRUE /* blocking */,
        current_offset /*offset: mode */, sizeof(double) /*size*/,
        &weight_value);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }
  RFASSERT_LOG("Weight upload complete!");
}

void RafkoGPUContext::set_network_weight(std::uint32_t weight_index,
                                         double weight_value) {
  RFASSERT_LOG("Setting weight[{}] to {}", weight_index, weight_value);
  RFASSERT(static_cast<std::int32_t>(weight_index) <
           m_network.weight_table_size());
  m_network.set_weight_table(weight_index, weight_value);
  m_solverFactory.refresh_actual_solution_weights();
  upload_weight_to_device(weight_index);
}

void RafkoGPUContext::set_network_weights(const std::vector<double> &weights) {
  RFASSERT_LOGV(weights, "Setting weights to:");
  RFASSERT(static_cast<std::int32_t>(weights.size()) ==
           m_network.weight_table_size());
  *m_network.mutable_weight_table() = {weights.begin(), weights.end()};
  m_solverFactory.refresh_actual_solution_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::apply_weight_update(
    const std::vector<double> &weight_delta) {
  RFASSERT_LOGV(weight_delta, "Applying weight update! Delta:");
  RFASSERT(static_cast<std::int32_t>(weight_delta.size()) ==
           m_network.weight_table_size());
  if (m_weightUpdater->is_finished())
    m_weightUpdater->start();
  m_weightUpdater->iterate(weight_delta);
  m_solverFactory.refresh_actual_solution_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::upload_weight_table_to_device() {
  RFASSERT_LOG("Uploading weight table to device..");
  std::vector<double> device_weight_table;
  std::uint32_t overall_number_of_weights = 0u;
  for (const rafko_net::PartialSolution &partial :
       m_solverFactory.actual_solution()->partial_solutions()) {
    device_weight_table.insert(device_weight_table.end(),
                               partial.weight_table().begin(),
                               partial.weight_table().end());
    overall_number_of_weights += partial.weight_table_size();
  }

  RFASSERT_LOGV(device_weight_table, "Weight table being uploaded to device:");
  RFASSERT(device_weight_table.size() == overall_number_of_weights);
  m_deviceWeightTableSize = device_weight_table.size();

  cl_int return_value = m_openclQueue.enqueueWriteBuffer(
      m_solutionPhase.get_input_buffer(), CL_TRUE /*blocking*/,
      sizeof(double) /*offset*/,
      (sizeof(double) * device_weight_table.size()) /*size*/,
      device_weight_table.data());
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
}

void RafkoGPUContext::refresh_objective() {
  RFASSERT_LOG("Refreshing objective in GPU context..");
  RFASSERT(static_cast<bool>(m_objective));
  m_objective->set_gpu_parameters(m_dataSet->get_number_of_label_samples(),
                                  m_dataSet->get_feature_size());
  RFASSERT_LOG("Updating Error Phase strategy: ");
  m_errorPhase.set_strategy(m_objective);
}

void RafkoGPUContext::set_objective(
    std::shared_ptr<rafko_gym::RafkoObjective> objective) {
  RFASSERT_LOG("Setting a new objective in GPU context");
  m_objective = objective;
  refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  m_lastRanEvaluation = not_eval_run;
}

void RafkoGPUContext::set_weight_updater(rafko_gym::Weight_updaters updater) {
  RFASSERT_LOG("Setting weight updater in GPU context to {}",
               rafko_gym::Weight_updaters_Name(updater));
  m_weightUpdater.reset();
  m_weightUpdater = rafko_gym::UpdaterFactory::build_weight_updater(
      m_network, updater, *m_settings);
}

void RafkoGPUContext::set_data_set(
    std::shared_ptr<rafko_gym::RafkoDataSet> data_set) {
  RFASSERT_SCOPE(ENV_BUILD);
  RFASSERT_LOG("Setting data set in GPU context..");
  RFASSERT_LOG(
      "Environment feature size: {} vs. Network output Neuron number: {}",
      data_set->get_feature_size(), m_network.output_neuron_number());
  RFASSERT(data_set->get_feature_size() == m_network.output_neuron_number());
  RFASSERT_LOG("Environment input size: {} vs. Network input size: {}",
               data_set->get_input_size(), m_network.input_data_size());
  RFASSERT(data_set->get_input_size() == m_network.input_data_size());

  m_dataSet.reset();
  m_dataSet = data_set;
  std::uint32_t old_output_buffer_num = m_neuronOutputsToEvaluate.size();
  std::uint32_t new_output_buffer_num =
      m_settings->get_max_processing_threads() *
          m_dataSet->get_sequence_size() +
      1u;
  m_neuronOutputsToEvaluate.resize(new_output_buffer_num);
  if (old_output_buffer_num < new_output_buffer_num) {
    for (std::uint32_t buffer_index = old_output_buffer_num - 1;
         buffer_index < new_output_buffer_num; ++buffer_index) {
      m_neuronOutputsToEvaluate[buffer_index].resize(
          m_dataSet->get_feature_size());
    }
  }
  m_neuronOutputsToEvaluate.back().resize(
      m_dataSet->get_number_of_label_samples());
  m_numOutputsInOneSequence = std::max(
      {2u, m_network.memory_size(), m_dataSet->get_inputs_in_one_sequence()});
  m_evalStartInSequence =
      m_numOutputsInOneSequence - m_dataSet->get_inputs_in_one_sequence();
  /*!Note: Network memory is not counting the "current" run; which is by
   * definition is not from memory */
  RFASSERT_LOG(
      "Agent sequence parameters: {} sequences; {} sequence_size; {} prefill "
      "inputs; {} Neurons",
      m_dataSet->get_number_of_sequences(), m_dataSet->get_sequence_size(),
      m_dataSet->get_prefill_inputs_number(), m_network.neuron_array_size());
  m_agent->set_sequence_params(m_dataSet->get_number_of_sequences(),
                               m_dataSet->get_sequence_size(),
                               m_dataSet->get_prefill_inputs_number());
  RFASSERT_LOG("Updating Solution Phase strategy: ");
  m_solutionPhase.set_strategy(m_agent);
  upload_weight_table_to_device();
  if (m_objective)
    refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  m_lastRanEvaluation = not_eval_run;
}

void RafkoGPUContext::upload_agent_output(
    std::uint32_t sequences_to_upload,
    std::uint32_t start_index_inside_sequence,
    std::uint32_t sequence_truncation,
    std::function<void(cl::Buffer, std::uint32_t, std::uint32_t)>
        data_handler) {
  RFASSERT_LOG(
      "Uploading agent outputs: inputs for a sequence: {}; slots for a "
      "sequence: {};  sequences to upload: {}; neuron number: {}; start index "
      "inside sequence: {}; sequence truncation: {}; network memory: {}",
      m_dataSet->get_inputs_in_one_sequence(), m_numOutputsInOneSequence,
      sequences_to_upload, m_network.neuron_array_size(),
      start_index_inside_sequence, sequence_truncation,
      m_network.memory_size());
  RFASSERT(0u < sequence_truncation);
  RFASSERT((start_index_inside_sequence + sequence_truncation) <=
           m_numOutputsInOneSequence);

  std::uint32_t src_byte_offset = 0u;
  const std::uint32_t network_output_size =
      (m_network.output_neuron_number() * sizeof(double));
  for (std::uint32_t sequence_index = 0; sequence_index < sequences_to_upload;
       ++sequence_index) {
    RFASSERT_LOG("Copying Agent data from Sequence[{} / {}]", sequence_index,
                 m_dataSet->get_number_of_sequences());
    for (std::uint32_t label_index = 0; label_index < m_numOutputsInOneSequence;
         ++label_index) {
      RFASSERT_LOG(
          "--> slot[{} / {}]; offset: {} --> {} + {}", label_index,
          m_numOutputsInOneSequence, src_byte_offset,
          (m_network.neuron_array_size() - m_network.output_neuron_number()) *
              sizeof(double),
          network_output_size);
      src_byte_offset +=
          ((m_network.neuron_array_size() - m_network.output_neuron_number()) *
           sizeof(double));
      if ((label_index >= start_index_inside_sequence) &&
          (label_index < (start_index_inside_sequence + sequence_truncation))) {
        RFASSERT_LOG("Copying Agent data from agent[{} + {} / {}];",
                     src_byte_offset, network_output_size,
                     m_agent->get_output_shapes()[0].get_byte_size<double>());
        RFASSERT((src_byte_offset + network_output_size) <=
                 m_agent->get_output_shapes()[0].get_byte_size<double>());
        data_handler(m_solutionPhase.get_output_buffer(), src_byte_offset,
                     network_output_size);
      }
      src_byte_offset += network_output_size;
    }
  }
}

double RafkoGPUContext::error_post_process(double raw_error,
                                           std::uint32_t labels_evaluated) {
  double error_value = raw_error;
  double divisor = std::max(labels_evaluated, 1u);
  double performance_error = m_solutionPhase.acquire_output(
      1u, m_agent->get_output_shapes()[0][0] /* first output, after the size of
                                                the first output */
      )[0];
  RFASSERT_LOG("Error post process: raw error value: {}; performance error: "
               "{}; divisor: {}",
               error_value, performance_error, divisor);
  return ((error_value + performance_error) / divisor);
}

double RafkoGPUContext::full_evaluation(bool force_gpu_upload) {
  [[maybe_unused]] cl_int return_value;
  std::vector<cl::Event> label_events;
  RFASSERT_SCOPE(GPU_FULL_EVALUATION);
  RFASSERT_LOG("Full evaluation in GPU Context..");
  RFASSERT(static_cast<bool>(m_objective));

  /* upload mode info */
  cl::Event fill_event;
  return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_solutionPhase.get_input_buffer(),
      0.0 /* mode value 0 means that network is under evaluation */,
      0u /*offset*/, sizeof(double) /*size(bytes)*/, NULL /*events to wit for*/,
      &fill_event);
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  return_value = fill_event.wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  if (force_gpu_upload || (m_lastRanEvaluation != full_eval_run)) {
    std::vector<cl::Event> input_events = m_dataSet->upload_inputs_to_buffer(
        m_openclQueue, m_solutionPhase.get_input_buffer(),
        sizeof(double) *
            (m_deviceWeightTableSize +
             m_agent->get_input_shapes()[0][0]) /*buffer_start_byte_offset*/,
        0u /*sequence_start_index*/, 0u /*buffer_sequence_start_index*/,
        m_dataSet->get_number_of_sequences() /*sequences_to_upload*/
    );

    RFASSERT_LOG("Number of Label samples in environment: {};",
                 m_dataSet->get_number_of_label_samples());
    label_events = m_dataSet->upload_labels_to_buffer(
        m_openclQueue, m_errorPhase.get_input_buffer(),
        (m_dataSet->get_number_of_label_samples() *
         m_dataSet->get_feature_size() *
         sizeof(double)) /*buffer_start_byte_offset*/,
        0u /*sequence_start_index*/, 0 /*buffer_sequence_start_index*/,
        m_dataSet->get_number_of_sequences() /*sequences_to_upload*/,
        0u /*start_index_inside_sequence*/,
        m_dataSet->get_sequence_size() /*sequence_truncation*/
    );

    for (cl::Event &input_event : input_events) {
      return_value = input_event.wait();
      if (CL_SUCCESS != return_value) {
        RFASSERT_LOG("OpenCL Return value: {}", return_value);
      }
      RFASSERT(return_value == CL_SUCCESS);
    }
  }

  /* run feature phase */
  m_solutionPhase(std::make_from_tuple<cl::EnqueueArgs>(
      std::tuple_cat(std::tie(m_openclQueue), m_agent->get_solution_space())));

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events;
  std::uint32_t dst_byte_offset = 0u;
  upload_agent_output(
      m_dataSet->get_number_of_sequences(),
      m_evalStartInSequence +
          m_dataSet
              ->get_prefill_inputs_number() /*start_index_inside_sequence*/,
      (m_numOutputsInOneSequence -
       (m_evalStartInSequence +
        m_dataSet->get_prefill_inputs_number())) /*sequence_truncation*/,
      [this, &features_events, &dst_byte_offset, &return_value](
          cl::Buffer source_buffer, std::uint32_t buffer_byte_offset,
          std::uint32_t data_byte_size) {
        RFASSERT_LOG("Copying from agent[{} + {} / {}] to objective[{} + {}] / "
                     "{} bytes; evaluation starts at slot: {}",
                     buffer_byte_offset, data_byte_size,
                     m_agent->get_output_shapes()[0].get_byte_size<double>(),
                     dst_byte_offset, data_byte_size,
                     m_objective->get_input_shapes()[0]
                         .get_byte_size<double>(), /* m_errorPhase is based on
                                                      m_objective */
                     m_evalStartInSequence);
        features_events.emplace_back();
        return_value =
            m_openclQueue
                .enqueueCopyBuffer(/* Upload sequence */
                                   source_buffer /*src*/,
                                   m_errorPhase.get_input_buffer() /*dst*/,
                                   buffer_byte_offset /*src_offset*/,
                                   dst_byte_offset /*dst_offset*/,
                                   data_byte_size /*size*/,
                                   NULL /*events to wait for*/,
                                   &features_events.back());
        if (CL_SUCCESS != return_value) {
          RFASSERT_LOG("OpenCL Return value: {}", return_value);
        }
        RFASSERT(return_value == CL_SUCCESS);
        dst_byte_offset += data_byte_size;
      });

  for (cl::Event &features_event : features_events) {
    return_value = features_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  for (cl::Event &label_event : label_events) {
    return_value = label_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  /* run error phase */
  m_errorPhase(std::make_from_tuple<cl::EnqueueArgs>(std::tuple_cat(
      std::tie(m_openclQueue), m_objective->get_solution_space())));

  RFASSERT_LOG("Last ran evaluation set to `Full evaluation run`");
  m_lastRanEvaluation = full_eval_run;
  return -error_post_process(m_errorPhase.acquire_output(1u)[0],
                             m_dataSet->get_number_of_label_samples());
}

double RafkoGPUContext::stochastic_evaluation(bool to_seed,
                                              std::uint32_t seed_value,
                                              bool force_gpu_upload) {
  [[maybe_unused]] cl_int return_value;
  std::vector<cl::Event> input_events;
  std::vector<cl::Event> label_events;
  RFASSERT_SCOPE(GPU_STOCHASTIC_EVALUATION);
  RFASSERT_LOG("Stochastic evaluation in GPU Context..");

  if (to_seed) {
    srand(seed_value);
    RFASSERT_LOG("Seeded run: last used seed: {}; current seed: {}",
                 m_lastUsedSeed, seed_value);
  }
  const std::uint32_t used_minibatch_size = std::min(
      m_settings->get_minibatch_size(), m_dataSet->get_number_of_sequences());
  const std::uint32_t used_sequence_truncation = std::min(
      m_settings->get_memory_truncation(), m_dataSet->get_sequence_size());
  const std::uint32_t start_index_inside_sequence =
      (rand() %
       (m_dataSet->get_sequence_size() - used_sequence_truncation + 1));
  RFASSERT_LOG("Used minibatch size: {}; sequence_truncation: {}; start index "
               "inside sequence: {}",
               used_minibatch_size, used_sequence_truncation,
               start_index_inside_sequence);
  if ((force_gpu_upload) || (m_lastRanEvaluation != random_eval_run) ||
      (m_lastUsedSeed != seed_value) || (!m_lastRandomEvalWasSeeded)) {
    cl::Event fill_event;
    RFASSERT_LOG("Updating evaluation buffer..");
    return_value =
        m_openclQueue
            .enqueueFillBuffer<double>(/* upload mode info */
                                       m_solutionPhase.get_input_buffer(),
                                       (0) /*the double value*/, 0u /*offset*/,
                                       sizeof(double) /*size(bytes)*/,
                                       NULL /*events to wit for*/, &fill_event);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
    return_value = fill_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);

    /* upload pseudo-random labels and inputs */
    std::uint32_t uploaded_sequences = 0u;
    while (uploaded_sequences < used_minibatch_size) {
      std::uint32_t sequences_to_upload =
          rand() % (used_minibatch_size - uploaded_sequences + 1u);
      std::uint32_t sequence_start_index =
          rand() %
          (m_dataSet->get_number_of_sequences() - sequences_to_upload + 1u);
      RFASSERT_LOG("Uploading {} sequences starting from {}",
                   sequences_to_upload, sequence_start_index);
      std::vector<cl::Event> input_events = m_dataSet->upload_inputs_to_buffer(
          m_openclQueue, m_solutionPhase.get_input_buffer(),
          sizeof(double) *
              (m_deviceWeightTableSize +
               m_agent->get_input_shapes()[0][0]) /*buffer_start_byte_offset*/,
          sequence_start_index,
          uploaded_sequences /*buffer_sequence_start_index*/,
          sequences_to_upload /*sequences_to_upload*/
      );
      input_events.insert(input_events.end(), input_events.begin(),
                          input_events.end());
      std::vector<cl::Event> label_events = m_dataSet->upload_labels_to_buffer(
          m_openclQueue, m_errorPhase.get_input_buffer(),
          (m_dataSet->get_number_of_label_samples() *
           m_dataSet->get_feature_size() *
           sizeof(double)) /*buffer_start_byte_offset*/,
          sequence_start_index,
          uploaded_sequences /*buffer_sequence_start_index*/,
          sequences_to_upload /*sequences_to_upload*/,
          start_index_inside_sequence, used_sequence_truncation);
      label_events.insert(label_events.end(), label_events.begin(),
                          label_events.end());
      uploaded_sequences += sequences_to_upload;
    } /*while(uploaded_sequences < used_minibatch_size)*/
  }

  if (to_seed) {
    m_lastUsedSeed = seed_value;
    m_lastRandomEvalWasSeeded = true;
  }

  for (cl::Event &event : input_events) {
    return_value = event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  /* run feature phase */
  std::tuple<cl::NDRange, cl::NDRange, cl::NDRange> sol_space =
      m_agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(used_minibatch_size);
  m_solutionPhase(std::make_from_tuple<cl::EnqueueArgs>(
      std::tuple_cat(std::tie(m_openclQueue), sol_space)));

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events;
  std::uint32_t dst_byte_offset = 0u;
  upload_agent_output(
      used_minibatch_size,
      m_evalStartInSequence + start_index_inside_sequence +
          m_dataSet->get_prefill_inputs_number(),
      used_sequence_truncation,
      [this, &features_events, &dst_byte_offset, &return_value](
          cl::Buffer source_buffer, std::uint32_t buffer_byte_offset,
          std::uint32_t data_byte_size) {
        RFASSERT_LOG("Copying from agent[{}] to objective[{} + {}] / {} bytes; "
                     "evaluation starts at slot: {}",
                     buffer_byte_offset, dst_byte_offset, data_byte_size,
                     m_objective->get_input_shapes()[0]
                         .get_byte_size<double>(), /* m_errorPhase is based on
                                                      m_objective */
                     m_evalStartInSequence);
        features_events.emplace_back();
        return_value =
            m_openclQueue
                .enqueueCopyBuffer(/* Upload sequence */
                                   source_buffer /*src*/,
                                   m_errorPhase.get_input_buffer() /*dst*/,
                                   buffer_byte_offset /*src_offset*/,
                                   dst_byte_offset /*dst_offset*/,
                                   data_byte_size /*size*/,
                                   NULL /*events to wait for*/,
                                   &features_events.back());
        if (CL_SUCCESS != return_value) {
          RFASSERT_LOG("OpenCL Return value: {}", return_value);
        }
        RFASSERT(return_value == CL_SUCCESS);
        dst_byte_offset += data_byte_size;
      });

  /* fill the rest of the output buffer with the label value */
  std::uint32_t uploaded_bytes_count =
      (m_dataSet->get_feature_size() * used_sequence_truncation *
       used_minibatch_size * sizeof(double));
  std::uint32_t minibatch_labels_byte_size =
      (m_dataSet->get_feature_size() * m_dataSet->get_sequence_size() *
       used_minibatch_size * sizeof(double));
  if (0 < (minibatch_labels_byte_size - uploaded_bytes_count)) {
    cl::Event fill_event;
    RFASSERT_LOG(
        "Copying label values[{}] to feature buffer[{} + {}] as dummy data..",
        minibatch_labels_byte_size, uploaded_bytes_count,
        (minibatch_labels_byte_size - uploaded_bytes_count));

    return_value = m_openclQueue.enqueueCopyBuffer(
        m_errorPhase.get_input_buffer() /*src*/,
        m_errorPhase.get_input_buffer() /*dst*/,
        minibatch_labels_byte_size /*src_offset*/,
        uploaded_bytes_count /*dst_offset*/,
        (minibatch_labels_byte_size - uploaded_bytes_count) /*size*/,
        NULL /*events to wait for*/, &fill_event);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);

    return_value = fill_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  for (cl::Event &features_event : features_events) {
    return_value = features_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }
  for (cl::Event &event : label_events) {
    return_value = event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  /* run error phase */
  std::tuple<cl::NDRange, cl::NDRange, cl::NDRange> error_sol_space =
      m_objective->get_solution_space();
  std::get<1>(error_sol_space) =
      cl::NDRange(used_minibatch_size * used_sequence_truncation);
  cl::EnqueueArgs error_enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
      std::tuple_cat(std::tie(m_openclQueue), error_sol_space));
  m_errorPhase(error_enque_arguments);
  RFASSERT_LOG("Last ran evaluation set to `Random evaluation run`");
  m_lastRanEvaluation = random_eval_run;
  return -error_post_process(
      m_errorPhase.acquire_output(1u)[0],
      (used_minibatch_size * m_dataSet->get_sequence_size()));
}

rafko_utilities::ConstVectorSubrange<>
RafkoGPUContext::solve(const std::vector<double> &input, bool reset_neuron_data,
                       std::uint32_t thread_index) {
  RFASSERT_SCOPE(GPU_STANDALONE_SOLVE);
  RFASSERT_LOG("Solving network in GPU Context..");
  RFASSERT_LOG("Thread index in solve: {}", thread_index);
  RFASSERT(0 == thread_index);
  if (0u != thread_index)
    throw std::runtime_error(
        "Multi-threaded openCL Environment not supported!");

  [[maybe_unused]] cl_int return_value;
  cl::Event fill_event;
  if (reset_neuron_data || (m_lastRanEvaluation != not_eval_run)) {
    RFASSERT_LOG("Resetting agent data..");
    return_value = m_openclQueue.enqueueFillBuffer<double>(
        m_solutionPhase.get_output_buffer(), 0.0 /* the data(pattern) value */,
        0u /*offset*/,
        (sizeof(double) * m_numOutputsInOneSequence *
         m_network.neuron_array_size()) /*size*/,
        NULL /*events to wait for*/, &fill_event);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);

    return_value = fill_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  /* upload mode info --> if the value is zero, then the Network is being
   * evaluated, not solved */
  return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_solutionPhase.get_input_buffer(), 1.0 /*mode_value*/, 0u /*offset*/,
      sizeof(double) /*size(bytes)*/, NULL /*events to wit for*/, &fill_event);
  /*!Note: mode value tells the Kernel to run the network one time for each
   * sequence ( given in dimensions ) and also that it is currently not being
   * evaluated.
   */
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  return_value = fill_event.wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  /* upload inputs */
  return_value = m_openclQueue.enqueueWriteBuffer(
      m_solutionPhase.get_input_buffer(), CL_TRUE,
      (sizeof(double) *
       (m_deviceWeightTableSize + 1u)) /*offset: mode and weights*/,
      (sizeof(double) * input.size()) /*size*/, input.data(), NULL);
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  std::tuple<cl::NDRange, cl::NDRange, cl::NDRange> sol_space =
      m_agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(1);
  m_solutionPhase(std::make_from_tuple<cl::EnqueueArgs>(
      std::tuple_cat(std::tie(m_openclQueue), sol_space)));

  /* upload agent output into provided output vector */
  std::uint32_t output_index = 0u;
  m_standaloneSolutionResult.resize(m_network.output_neuron_number());
  upload_agent_output(
      1u /*sequences_to_upload*/, 0u /*start_index_inside_sequence*/,
      m_numOutputsInOneSequence /*sequence_truncation*/,
      [this, &output_index, &return_value,
       &fill_event](cl::Buffer source_buffer, std::uint32_t buffer_byte_offset,
                    std::uint32_t data_byte_size) {
        static const std::uint32_t relevant_feature_index =
            m_numOutputsInOneSequence - 1u;
        if (output_index == relevant_feature_index) {
          RFASSERT_LOG("copying from agent[{}] to buffer: {} / {} bytes..",
                       buffer_byte_offset, data_byte_size,
                       (m_standaloneSolutionResult.size() * sizeof(double)));
          return_value = m_openclQueue.enqueueReadBuffer(
              source_buffer /*src*/, CL_TRUE /*blocking*/, buffer_byte_offset,
              data_byte_size /*size*/, m_standaloneSolutionResult.data(),
              NULL /*events to wit for*/, &fill_event);
          if (CL_SUCCESS != return_value) {
            RFASSERT_LOG("OpenCL Return value: {}", return_value);
          }
          RFASSERT(return_value == CL_SUCCESS);
        }
        ++output_index;
      });
  // std::cout << "agent output: ";
  // for(double d : m_standaloneSolutionResult) std::cout << "[" << d << "]";
  // std::cout << std::endl;
  return_value = fill_event.wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  m_lastRanEvaluation = not_eval_run;
  return {m_standaloneSolutionResult.begin(), m_standaloneSolutionResult.end()};
}

void RafkoGPUContext::solve_data_set(std::vector<std::vector<double>> &output,
                                     bool isolated) {
  RFASSERT(output.size() == (m_dataSet->get_number_of_sequences() *
                             m_dataSet->get_sequence_size()));
  [[maybe_unused]] cl_int return_value;

  /* upload mode info */
  cl::Event fill_event;
  return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_solutionPhase.get_input_buffer(),
      static_cast<double>(
          m_dataSet->get_inputs_in_one_sequence()) /*mode_value*/,
      0u /*offset*/, sizeof(double) /*size(bytes)*/, NULL /*events to wit for*/,
      &fill_event);
  /*!Note: mode value tells the network that it's not being evaluated and to run
   * the network *value* times for one sequence */
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);
  return_value = fill_event.wait();
  if (CL_SUCCESS != return_value) {
    RFASSERT_LOG("OpenCL Return value: {}", return_value);
  }
  RFASSERT(return_value == CL_SUCCESS);

  if (isolated) {
    RFASSERT_LOG("Resetting agent data..");
    return_value = m_openclQueue.enqueueFillBuffer<double>(
        m_solutionPhase.get_output_buffer(), 0.0 /* the data(pattern) value */,
        0u /*offset*/,
        (sizeof(double) * m_numOutputsInOneSequence *
         m_network.neuron_array_size() *
         m_dataSet->get_number_of_sequences()) /*size*/,
        NULL /*events to wait for*/, &fill_event);
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  if (m_lastRanEvaluation != full_eval_run) { /* The same data is needed on the
                                                 GPU for full evaluation */
    std::vector<cl::Event> input_events = m_dataSet->upload_inputs_to_buffer(
        m_openclQueue, m_solutionPhase.get_input_buffer(),
        sizeof(double) *
            (m_deviceWeightTableSize +
             m_agent->get_input_shapes()[0][0]) /*buffer_start_byte_offset*/,
        0u /*sequence_start_index*/, 0u /*buffer_sequence_start_index*/,
        m_dataSet->get_number_of_sequences() /*sequences_to_upload*/
    );

    for (cl::Event &input_event : input_events) {
      return_value = input_event.wait();
      if (CL_SUCCESS != return_value) {
        RFASSERT_LOG("OpenCL Return value: {}", return_value);
      }
      RFASSERT(return_value == CL_SUCCESS);
    }
  }

  /* run feature phase */
  m_solutionPhase(std::make_from_tuple<cl::EnqueueArgs>(
      std::tuple_cat(std::tie(m_openclQueue), m_agent->get_solution_space())));

  /* upload agent output into provided output vector */
  std::vector<cl::Event> features_events;
  std::uint32_t raw_label_index = 0u;
  std::uint32_t output_index = 0u;
  upload_agent_output(
      m_dataSet->get_number_of_sequences(), 0u /*start_index_inside_sequence*/,
      m_dataSet->get_sequence_size() /*sequence_truncation*/,
      [this, &features_events, &output, &output_index, &raw_label_index,
       &return_value](cl::Buffer source_buffer,
                      std::uint32_t buffer_byte_offset,
                      std::uint32_t data_byte_size) {
        if ((raw_label_index % m_numOutputsInOneSequence) >=
            m_evalStartInSequence) {
          RFASSERT_LOG("copying from agent_buf[{}] label[{} / {}] to "
                       "output[{}]; {} / {} bytes..",
                       buffer_byte_offset, raw_label_index,
                       (m_dataSet->get_number_of_sequences() *
                        m_numOutputsInOneSequence),
                       output_index, buffer_byte_offset, data_byte_size,
                       (output[output_index].size() * sizeof(double)));
          RFASSERT(output_index < output.size());
          RFASSERT(output[output_index].size() ==
                   (data_byte_size / sizeof(double)));
          features_events.emplace_back();
          return_value = m_openclQueue.enqueueReadBuffer(
              source_buffer /*src*/, CL_FALSE /*blocking*/,
              buffer_byte_offset /*src_offset*/, data_byte_size /*size*/,
              output[output_index].data(), NULL /*events to wait for*/,
              &features_events.back());
          if (CL_SUCCESS != return_value) {
            RFASSERT_LOG("OpenCL Return value: {}", return_value);
          }
          RFASSERT(return_value == CL_SUCCESS);

          ++output_index;
        }
        ++raw_label_index;
      });

  for (cl::Event &features_event : features_events) {
    return_value = features_event.wait();
    if (CL_SUCCESS != return_value) {
      RFASSERT_LOG("OpenCL Return value: {}", return_value);
    }
    RFASSERT(return_value == CL_SUCCESS);
  }

  /* In case the last run was full evaluation, no data is changed on the GPU
   * with this run.. */
  if (m_lastRanEvaluation !=
      full_eval_run) /* ..so the state flag can remain on its previous state. */
    m_lastRanEvaluation = not_eval_run;
}

} /* namespace rafko_mainframe */
