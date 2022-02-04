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

#include "rafko_mainframe/services/rafko_gpu_context.h"

#include <assert.h>

#include "rafko_protocol/solution.pb.h"
#include "rafko_net/services/solution_builder.h"
#include "rafko_mainframe/services/rafko_dummies.h"

namespace rafko_mainframe{

RafkoGPUContext::Builder::Builder(rafko_net::RafkoNet neural_network_, rafko_mainframe::RafkoSettings settings_)
: settings(settings_)
, network(neural_network_)
{
  cl::Platform::get(&platforms);
  assert( 0 < platforms.size() );
}

RafkoGPUContext::Builder& RafkoGPUContext::Builder::select_platform(uint32 platform_index){
  assert( platform_index < platforms.size() );
  selected_platform = platform_index;
  return *this;
}

RafkoGPUContext::Builder& RafkoGPUContext::Builder::select_device(cl_device_type type, uint32 device_index){
  platforms[selected_platform].getDevices(type, &devices);
  assert( device_index < devices.size() );
  selected_device = device_index;
  return *this;
}

std::unique_ptr<RafkoGPUContext> RafkoGPUContext::Builder::build(){
  assert( 0 < platforms.size() );
  assert( 0 < devices.size() );
  cl::Context context({devices[selected_device]});
  return std::unique_ptr<RafkoGPUContext>( new RafkoGPUContext(
    context, devices[selected_device], std::move(settings), std::move(network)
  ) );
}

RafkoGPUContext::RafkoGPUContext(
  cl::Context& context_, cl::Device& device_,
  rafko_mainframe::RafkoSettings&& settings_, rafko_net::RafkoNet&& neural_network_
):settings(settings_.set_arena_ptr(&arena))
, network(neural_network_)
, network_solution(rafko_net::SolutionBuilder(settings).build(network))
, agent(rafko_net::SolutionSolver::Builder(*network_solution, settings).build())
, environment(std::make_unique<RafkoDummyEnvironment>(
  network.input_data_size(), network.output_neuron_number())
),objective(std::make_unique<RafkoDummyObjective>())
, weight_updater(rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, rafko_gym::weight_updater_amsgrad, settings))
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * environment->get_sequence_size() + 1u),
  std::vector<sdouble32>(network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
),execution_threads(settings.get_max_processing_threads())
, opencl_context(context_)
, opencl_device(device_)
, opencl_queue(opencl_context, opencl_device)
, solution_phase( opencl_context, opencl_device, opencl_queue, agent )
, error_phase(
  opencl_context, opencl_device, opencl_queue,
  std::make_shared<RafkoDummyGPUStrategyPhase>(
    RafkoNBufShape({network.output_neuron_number(), network.output_neuron_number()}),
    RafkoNBufShape({1u})
  )
),used_sequence_truncation( std::min(settings.get_memory_truncation(), environment->get_sequence_size()) )
, used_minibatch_size( std::min(settings.get_minibatch_size(), environment->get_number_of_sequences()) )
{
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  upload_weight_table_to_device(); /*!Note: Also sets device_weight_table_size*/
  refresh_objective();
}

void RafkoGPUContext::set_network_weight(uint32 weight_index, sdouble32 weight_value){
  assert( static_cast<sint32>(weight_index) < network.weight_table_size() );
  network.set_weight_table(weight_index, weight_value);
  weight_updater->update_solution_with_weights();
  upload_weight_table_to_device(); /* TODO: modify single weight  */
}

void RafkoGPUContext::set_network_weights(const std::vector<sdouble32>& weights){
  assert( static_cast<sint32>(weights.size()) == network.weight_table_size() );
  *network.mutable_weight_table() = {weights.begin(), weights.end()};
  weight_updater->update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::apply_weight_update(const std::vector<sdouble32>& weight_delta){
  assert( static_cast<sint32>(weight_delta.size()) == network.weight_table_size() );
  if(weight_updater->is_finished())
    weight_updater->start();
  weight_updater->iterate(weight_delta);
  weight_updater->update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::upload_weight_table_to_device(){
  std::vector<sdouble32> device_weight_table;
  uint32 overall_number_of_weights = 0u;
  std::cout << "solution weight table: ";
  for(const rafko_net::PartialSolution& partial : network_solution->partial_solutions()){
    for(const sdouble32& w : partial.weight_table())std::cout << "{" << w << "}";
    std::cout << "|";
    device_weight_table.insert(
      device_weight_table.end(),
      partial.weight_table().begin(), partial.weight_table().end()
    );
    overall_number_of_weights += partial.weight_table_size();
  }
  std::cout << std::endl;

  assert( device_weight_table.size() == overall_number_of_weights );
  device_weight_table_size = device_weight_table.size();

  cl_int return_value = opencl_queue.enqueueWriteBuffer(
    solution_phase.get_input_buffer(), CL_TRUE/*blocking*/, sizeof(sdouble32)/*offset*/,
    (sizeof(sdouble32) * device_weight_table.size())/*size*/,
    device_weight_table.data()
  );
  assert( return_value == CL_SUCCESS );
}

void RafkoGPUContext::refresh_objective(){
  objective->set_gpu_parameters(
    environment->get_number_of_label_samples(),
    environment->get_feature_size()
  );
  error_phase.set_strategy(objective);
  std::cout << "features and labels element size: " << objective->get_input_shapes()[0].get_number_of_elements() << std::endl;
}

void RafkoGPUContext::set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_){
  objective = objective_;
  refresh_objective();
}

void RafkoGPUContext::set_weight_updater(rafko_gym::Weight_updaters updater){
  weight_updater = rafko_gym::UpdaterFactory::build_weight_updater(network, *network_solution, updater, settings);
}

void RafkoGPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_){
  assert(environment_->get_feature_size() == network.output_neuron_number());
  assert(environment_->get_input_size() == network.input_data_size());
  environment.reset();
  environment = environment_;
  uint32 old_output_buffer_num = neuron_outputs_to_evaluate.size();
  uint32 new_output_buffer_num = settings.get_max_processing_threads() * environment->get_sequence_size() + 1u;
  neuron_outputs_to_evaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(uint32 buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      neuron_outputs_to_evaluate[buffer_index].resize(environment->get_feature_size());
    }
  }
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  used_sequence_truncation = std::min(settings.get_memory_truncation(), environment->get_sequence_size());
  used_minibatch_size = std::min(settings.get_minibatch_size(), environment->get_number_of_sequences());
  agent->set_sequence_params(
    environment->get_number_of_sequences(),
    environment->get_sequence_size(),
    environment->get_prefill_inputs_number()
  );
  solution_phase.set_strategy(agent);
  upload_weight_table_to_device();
  refresh_objective();
  last_ran_evaluation = not_eval_run;
}

std::vector<cl::Event> RafkoGPUContext::upload_agent_inputs(
  uint32 sequence_start_index, uint32 buffer_sequence_start_index, uint32 sequences_to_upload
){
  cl_int return_value;
  uint32 elements_in_a_sequence = environment->get_sequence_size() + environment->get_prefill_inputs_number();
  /*!Note: elements == inputs */
  uint32 raw_input_start = (sequence_start_index * elements_in_a_sequence);
  uint32 raw_input_num = (sequences_to_upload * elements_in_a_sequence);
  uint32 input_buffer_byte_offset = (
    (
      (device_weight_table_size + 1u)
      + (buffer_sequence_start_index * elements_in_a_sequence * environment->get_input_size())
    ) * sizeof(sdouble32)
  );
  // std::cout << "device_weight_table_size:" << device_weight_table_size << std::endl;
  // std::cout << "buffer_sequence_start_index:" << buffer_sequence_start_index << std::endl;
  // std::cout << "elements_in_a_sequence:" << elements_in_a_sequence << std::endl;
  // std::cout << "environment->get_input_size():" << environment->get_input_size() << std::endl;
  // std::cout << "input_buffer_byte_offset:" << input_buffer_byte_offset << std::endl;
  // std::cout << "vs input buffer size:" << agent->get_input_shapes()[0].get_byte_size<sdouble32>() << std::endl;

  std::vector<cl::Event> events(raw_input_num);

  assert( (raw_input_start + raw_input_num) <= environment->get_number_of_input_samples() );
  // std::cout << "upl inputs:";
  for(uint32 raw_input_index = raw_input_start; raw_input_index < (raw_input_start + raw_input_num); ++raw_input_index){
    // std::cout << "input_buffer_byte_offset:" << input_buffer_byte_offset << std::endl;
    // std::cout << "\t - ";
    // for(const sdouble32& input : environment->get_input_sample(raw_input_index)){
    //   std::cout << "["<< input << "]";
    // }
    return_value = opencl_queue.enqueueWriteBuffer(
      solution_phase.get_input_buffer(), CL_FALSE/*blocking*/,
      input_buffer_byte_offset/*offset*/,
      (sizeof(sdouble32) * environment->get_input_sample(raw_input_index).size())/*size*/,
      environment->get_input_sample(raw_input_index).data(),
      NULL, &events[raw_input_index - raw_input_start]
    );
    assert( return_value == CL_SUCCESS );

    input_buffer_byte_offset += (sizeof(sdouble32) * environment->get_input_sample(raw_input_index).size());
  }
  return events;
}

std::vector<cl::Event> RafkoGPUContext::upload_labels(
  uint32 sequence_start_index, uint32 buffer_sequence_start_index,
  uint32 sequences_to_upload, uint32 buffer_start_byte_offset
){
  cl_int return_value;
  uint32 elements_in_a_sequence = environment->get_sequence_size();
  /*!Note: elements == labels */
  uint32 raw_label_start = (sequence_start_index * elements_in_a_sequence);
  uint32 raw_label_num = (sequences_to_upload * elements_in_a_sequence);
  uint32 buffer_byte_offset = (
    buffer_start_byte_offset
    + (
      buffer_sequence_start_index * elements_in_a_sequence
      * environment->get_feature_size() * sizeof(sdouble32)
    )
  );

  assert( (raw_label_start + raw_label_num) <= environment->get_number_of_label_samples() );

  std::cout << "label samples:\t -";
  uint32 labels_byte_offset = 0u;
  std::vector<cl::Event> events(raw_label_num);
  for(uint32 raw_label_index = raw_label_start; raw_label_index < (raw_label_start + raw_label_num); ++raw_label_index){
    uint32 label_byte_size =  (sizeof(sdouble32) * environment->get_label_sample(raw_label_index).size());
    for(const sdouble32& label_element : environment->get_label_sample(raw_label_index))std::cout << "["<< label_element << "]\t";
    std::cout << "offset+size: (" << buffer_byte_offset << "+" << labels_byte_offset
    << ") + " << label_byte_size << " / " << objective->get_input_shapes()[0].get_byte_size<sdouble32>()
    << std::endl;
    return_value = opencl_queue.enqueueWriteBuffer(
      error_phase.get_input_buffer(), CL_FALSE/*blocking*/,
      (buffer_byte_offset + labels_byte_offset)/*offset*/,
      label_byte_size/*size*/, environment->get_label_sample(raw_label_index).data(),
      NULL, &events[raw_label_index - raw_label_start]
    );
    assert( return_value == CL_SUCCESS );
    labels_byte_offset += label_byte_size;
    std::cout << "\t - ";
  }
  std::cout << std::endl;
  return events;
}

std::vector<cl::Event> RafkoGPUContext::upload_agent_output(uint32 sequences_to_upload){
  cl_int return_value;
  uint32 elements_to_upload = (sequences_to_upload * environment->get_sequence_size());
  /*!Note: elements == features */
  uint32 byte_offset_solution_phase = 0u;
  uint32 byte_offset_error_phase = 0u;
  std::vector<cl::Event> events(elements_to_upload);
  std::cout << "sizes:"
  << (
    elements_to_upload
    * (environment->get_sequence_size() + environment->get_prefill_inputs_number())
    * environment->get_feature_size()
    * sizeof(sdouble32)
  )
  << "<>" << (objective->get_input_shapes()[0].get_byte_size<sdouble32>())
  << "<>" << (agent->get_output_shapes()[0].get_byte_size<sdouble32>())
  << std::endl;
  std::cout << "environment->get_prefill_inputs_number(): " << environment->get_prefill_inputs_number() << std::endl;
  std::cout << "network.neuron_array_size(): " << network.neuron_array_size() << std::endl;
  std::cout << "Uploading result...";
  for(uint32 sequence_index = 0; sequence_index < sequences_to_upload; ++sequence_index){
    std::cout << "\ns("<< sequence_index<<")";
    byte_offset_solution_phase += (environment->get_prefill_inputs_number() * network.neuron_array_size() * sizeof(sdouble32));
    // std::cout << "prf(" <<
    // (environment->get_prefill_inputs_number() * network.output_neuron_number() * sizeof(sdouble32))
    // << ")";
    for(uint32 feature_index = 0; feature_index < environment->get_sequence_size(); ++feature_index){
      std::cout << "f("<< feature_index<<";";
      std::cout << "offsets: "<< byte_offset_error_phase <<"; "<< byte_offset_solution_phase <<")";
      /* add neuron_array_offset */
      byte_offset_solution_phase += ( (network.neuron_array_size() - network.output_neuron_number()) * sizeof(sdouble32) );
      /* Upload sequence size */
      return_value = opencl_queue.enqueueCopyBuffer(
        solution_phase.get_output_buffer() /*src*/, error_phase.get_input_buffer() /*dst*/,
        byte_offset_solution_phase/*src_offset*/, byte_offset_error_phase/*dst_offset*/,
        (network.output_neuron_number() * sizeof(sdouble32))/*size*/,
        NULL /*events to wait for*/, &events[(sequence_index * environment->get_sequence_size()) + feature_index]
      );
      assert( return_value == CL_SUCCESS );

      /*!Debug: see what is actually being uploaded */
      std::vector<sdouble32> sequence_content(environment->get_feature_size());
      return_value = opencl_queue.enqueueReadBuffer( /* download last output from device memory */
        solution_phase.get_output_buffer(), CL_TRUE/*blocking*/,
        byte_offset_solution_phase/*offset*/, (sizeof(sdouble32) * environment->get_feature_size())/*size*/,
        static_cast<void*>(sequence_content.data())
      );
      assert( return_value == CL_SUCCESS );
      for(const sdouble32& s : sequence_content)std::cout << "[" << s << "]";
      std::cout << " - ";
      byte_offset_solution_phase += (network.output_neuron_number() * sizeof(sdouble32));
      byte_offset_error_phase += (network.output_neuron_number() * sizeof(sdouble32));
    }
    std::cout << "s(/) ";
  }
  std::cout << std::endl;
  return events;
}

sdouble32 RafkoGPUContext::full_evaluation(){
  cl_int return_value;
  std::vector<cl::Event> label_events;

  // std::cout << "sequence_num: " << environment->get_number_of_sequences() << std::endl;
  // std::cout << "sequence_size: " << environment->get_sequence_size() << std::endl;
  // std::cout << "prefills: " << environment->get_prefill_inputs_number() << std::endl;
  // std::cout << "mem: " << network_solution->network_memory_length() << std::endl;
  if(last_ran_evaluation != full_eval_run){
    /* upload mode info */
    cl::Event fill_event;
    return_value = opencl_queue.enqueueFillBuffer<sdouble32>(
      solution_phase.get_input_buffer(), double_literal(0.0)/*the sdouble32 value*/,
      0u /*offset*/, sizeof(sdouble32)/*size(bytes)*/,
      NULL/*events to wit for*/, &fill_event
    );
    assert( return_value == CL_SUCCESS );
    return_value = fill_event.wait();
    assert( return_value == CL_SUCCESS );

    std::vector<cl::Event> input_events = upload_agent_inputs(
      0u/*sequence_start_index*/, 0u/*buffer_sequence_start_index*/,
      environment->get_number_of_sequences()/*sequences_to_upload*/
    );

    label_events = upload_labels(
      0u/*sequence_start_index*/, 0/*buffer_sequence_start_index*/,
      environment->get_number_of_sequences()/*sequences_to_upload*/,
      (environment->get_number_of_label_samples() * environment->get_feature_size() * sizeof(sdouble32)) /*buffer_start_byte_offset*/
    );

    for(cl::Event& input_event : input_events){
      return_value = input_event.wait();
      assert( return_value == CL_SUCCESS );
    }
  }

  // /*!Debug: see if the input data is uploaded correctly */
  // std::vector<sdouble32> full_uploaded_input(agent->get_input_shapes()[0].get_number_of_elements());
  // return_value = opencl_queue.enqueueReadBuffer( /* download last output from device memory */
  //   solution_phase.get_input_buffer(), CL_TRUE/*blocking*/,
  //   0/*offset*/, agent->get_input_shapes()[0].get_byte_size<sdouble32>()/*size*/,
  //   static_cast<void*>(full_uploaded_input.data())
  // );
  // assert( return_value == CL_SUCCESS );
  //
  // std::cout << "full input:";
  // for(const sdouble32& in : full_uploaded_input)std::cout << "[" << in << "]";
  // std::cout << std::endl;

  // /*!Debug: see if the output data is reseted correctly */
  // std::vector<sdouble32> supposed_out(agent->get_input_shapes()[0].get_number_of_elements());
  // return_value = opencl_queue.enqueueReadBuffer( /* download last output from device memory */
  //   solution_phase.get_output_buffer(), CL_TRUE/*blocking*/,
  //   0/*offset*/, agent->get_output_shapes()[0].get_byte_size<sdouble32>()/*size*/,
  //   static_cast<void*>(supposed_out.data())
  // );
  // assert( return_value == CL_SUCCESS );
  //
  // std::cout << "output should be all zeroes:";
  // for(const sdouble32& in : supposed_out)std::cout << "[" << in << "]";
  // std::cout << std::endl;

  /* run feature phase */
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), agent->get_solution_space())
  );
  solution_phase( enque_arguments );

  std::vector<cl::Event> features_events = upload_agent_output(environment->get_number_of_sequences());

  for(cl::Event& features_event : features_events){
    return_value = features_event.wait();
    assert( return_value == CL_SUCCESS );
  }

  for(cl::Event& label_event : label_events){
    return_value = label_event.wait();
    assert( return_value == CL_SUCCESS );
  }

  // /*!Debug: check the calculated features */
  // // std::cout << "agent output size: " << agent->get_output_shapes()[0].get_number_of_elements() << std::endl;
  // std::vector<sdouble32> agent_output(agent->get_output_shapes()[0].get_number_of_elements());
  // return_value = opencl_queue.enqueueReadBuffer( /* download last output from device memory */
  //   solution_phase.get_output_buffer(), CL_TRUE/*blocking*/,
  //   0/*offset*/, agent->get_output_shapes()[0].get_byte_size<sdouble32>()/*size*/,
  //   static_cast<void*>(agent_output.data())
  // );
  // assert( return_value == CL_SUCCESS );
  // std::cout << "Agent output:";
  // uint32 out_i = 0;
  // for(const sdouble32& num : agent_output){
  //   if(0 == (out_i % network.neuron_array_size())){
  //     std::cout << "\n - ";// << std::endl;
  //     // for(int i = 0; i<out_i;++i)std::cout << " ";
  //   }
  //    std::cout << "["<<num<<"]";
  //   ++out_i;
  // }
  // std::cout << std::endl;
  //
  // std::vector<sdouble32> uploaded_features_labels(objective->get_input_shapes()[0].get_number_of_elements());
  // return_value = opencl_queue.enqueueReadBuffer( /* download last output from device memory */
  //   error_phase.get_input_buffer(), CL_TRUE/*blocking*/,
  //   0/*offset*/, objective->get_input_shapes()[0].get_byte_size<sdouble32>()/*size*/,
  //   static_cast<void*>(uploaded_features_labels.data())
  // );
  // assert( return_value == CL_SUCCESS );
  // std::cout << "Actual Uploaded features and labels:";
  // out_i = 0;
  // for(const sdouble32& num : uploaded_features_labels){
  //   if(0 == (out_i % environment->get_feature_size())) std::cout << "\n - ";// << std::endl;
  //   std::cout << "["<<num<<"]";
  //   ++out_i;
  // }
  // std::cout << std::endl;

  // sdouble32 manual_result = 0.0;
  // for(uint32 i = 0; i < uploaded_features_labels.size()/2; ++i){
  //   manual_result += std::pow(
  //     uploaded_features_labels[i] - uploaded_features_labels[objective->get_input_shapes()[0][0] + i],
  //     2.0
  //   ) / 2.0;
  // }
  // std::cout << "manual_result: " << (manual_result/environment->get_number_of_label_samples()) << std::endl;
  //
  // std::cout << "with structure(" << objective->get_input_shapes().size() << "): ";
  // std::cout << "[" << objective->get_input_shapes()[0][0] <<"]";
  // std::cout << "[" << objective->get_input_shapes()[0][1] <<"]";
  // std::cout << std::endl;

  /* run error phase */
  cl::EnqueueArgs error_enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), objective->get_solution_space())
  );
  error_phase( error_enque_arguments );

  /* get resulting error */
  last_ran_evaluation = full_eval_run;
  // std::cout << "GPU Error sum: " << error_phase.acquire_output(1u)[0] << std::endl;
  return -(error_phase.acquire_output(1u)[0] / environment->get_number_of_label_samples());
}

sdouble32 RafkoGPUContext::stochastic_evaluation(bool to_seed, uint32 seed_value){
  if(to_seed)srand(seed_value);
  #warning "stochastic_evaluation not implemented yet"
  /* TODO: Neuron outputs size depend on the network memory, which might be bigger, than the sequence or the minibatch! */
  /* TODO: Keep sequence truncation in mind  */
  return 0;
}

rafko_utilities::ConstVectorSubrange<> RafkoGPUContext::solve(
  const std::vector<sdouble32>& input,
  bool reset_neuron_data, uint32 thread_index
){
  if(0u != thread_index)
    throw std::runtime_error("Multi-threaded openCL Environment not supported!");

  cl_int return_value;
  cl::Event fill_event;

  if(reset_neuron_data){
    return_value = opencl_queue.enqueueFillBuffer<sdouble32>(
      solution_phase.get_output_buffer(), double_literal(0.0)/* the data(pattern) value */,
      0u/*offset*/, agent->get_output_shapes()[0][0]/*size == number of sdouble32*/,
      NULL/*events to wait for*/, &fill_event
    );
    assert( return_value == CL_SUCCESS );
    return_value = fill_event.wait();
    assert( return_value == CL_SUCCESS );
  }else{ /* Neuron memory not resetted, keep network memory consistent */
    for(uint32 memory_slot = 0; memory_slot < (network_solution->network_memory_length() - 1u); ++memory_slot){
      uint32 network_memory_span_bytes = network_solution->neuron_number() * sizeof(sdouble32);
      return_value = opencl_queue.enqueueCopyBuffer(
        solution_phase.get_output_buffer()/*src*/, solution_phase.get_output_buffer() /*dst*/,
        ((memory_slot + 1u) * network_memory_span_bytes)/*src_offset*/,
        (memory_slot * network_memory_span_bytes)/*dst_offset*/,
        network_memory_span_bytes/*size*/, NULL, &fill_event
      );
      assert( return_value == CL_SUCCESS );
      return_value = fill_event.wait();
      assert( return_value == CL_SUCCESS );
    }
  }

  /* upload mode info */
  return_value = opencl_queue.enqueueFillBuffer<sdouble32>(
    solution_phase.get_input_buffer(), double_literal(69.420)/*the sdouble32 value*/,
    0u /*offset*/, sizeof(sdouble32)/*size(bytes)*/,
    NULL/*events to wit for*/, &fill_event
  );
  assert( return_value == CL_SUCCESS );
  return_value = fill_event.wait();
  assert( return_value == CL_SUCCESS );

  /* upload inputs */
  return_value = opencl_queue.enqueueWriteBuffer(
    solution_phase.get_input_buffer(), CL_TRUE,
    (sizeof(sdouble32) * (device_weight_table_size + 1u))/*offset: mode and weights*/,
    (sizeof(sdouble32) * input.size())/*size*/,
    input.data(), NULL
  );
  assert( return_value == CL_SUCCESS );

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> sol_space = agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(1);
  cl::EnqueueArgs enq = std::make_from_tuple<cl::EnqueueArgs>( std::tuple_cat(std::tie(opencl_queue), sol_space) );
  solution_phase( enq );

  uint32 output_array_start = ( /* the end of the last memory slot contains the network data */
    (network_solution->network_memory_length() * network.neuron_array_size()) - network.output_neuron_number()
  );
  if(static_cast<sint32>(standalone_solution_result.size()) == network.neuron_array_size()){
    solution_phase.load_output(standalone_solution_result.data(), network.output_neuron_number(), output_array_start);
  }else{
    std::unique_ptr<sdouble32[]> output_ptr = solution_phase.acquire_output(network.output_neuron_number(), output_array_start);
    standalone_solution_result = std::vector<sdouble32>(output_ptr.get(), output_ptr.get() + network.output_neuron_number());
  }

  std::cout << "End result in single run:";
  for(const sdouble32& num : standalone_solution_result)std::cout << "["<< num <<"]";
  std::cout << std::endl;

  std::unique_ptr<sdouble32[]> full_output_ptr = solution_phase.acquire_output(
    agent->get_output_shapes()[0].get_number_of_elements()
  );
  std::vector<sdouble32> full_output = std::vector<sdouble32>(
    full_output_ptr.get(), full_output_ptr.get() + agent->get_output_shapes()[0].get_number_of_elements()
  );
  std::cout << "Full output:";
  for(const sdouble32& num : full_output)std::cout << "["<< num <<"]";
  std::cout << std::endl;

  last_ran_evaluation = not_eval_run;

  return { standalone_solution_result.end() - network.output_neuron_number(), standalone_solution_result.end() };
}

} /* namespace rafko_mainframe */
