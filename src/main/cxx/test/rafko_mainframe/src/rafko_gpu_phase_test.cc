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
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <memory>

#include "rafko_mainframe/models/rafko_gpu_strategy.hpp"
#include "rafko_mainframe/services/rafko_gpu_phase.hpp"
#include "rafko_mainframe/services/rafko_ocl_factory.hpp"

#include "test/test_utility.hpp"

namespace rafko_mainframe_test {

class EchoStrategy : public rafko_mainframe::RafkoGPUStrategy {
public:
  EchoStrategy(std::uint32_t content_count = 5)
      : m_contentCount(content_count) {}

  std::vector<std::string> get_step_names() const { return {"echoes"}; }

  cl::Program::Sources get_step_sources() const {
    return {R"(
    void __kernel echoes(
       __constant double* inputs, __constant int* input_sizes, int input_sizes_size,
       __global double* outputs, __constant int* output_sizes, int output_sizes_size
    ){
      int index = get_global_id(0);
      if(index < output_sizes[0]){
        outputs[index] = inputs[index];
      }
    }
    )"};
  }

  std::vector<rafko_mainframe::RafkoNBufShape> get_input_shapes() const {
    return {rafko_mainframe::RafkoNBufShape{m_contentCount}};
  }
  std::vector<rafko_mainframe::RafkoNBufShape> get_output_shapes() const {
    return {rafko_mainframe::RafkoNBufShape{m_contentCount}};
  }
  std::tuple<cl::NDRange, cl::NDRange, cl::NDRange> get_solution_space() const {
    return {cl::NullRange, cl::NDRange(m_contentCount), cl::NullRange};
  }

private:
  std::uint32_t m_contentCount;
};

TEST_CASE("Testing if the simplest OpenCL Implementation is working as "
          "expected with std::vector input",
          "[GPU][Phase][simple]") {
  constexpr const std::size_t element_count = 10;
  std::shared_ptr<EchoStrategy> strategy =
      std::make_unique<EchoStrategy>(element_count);
  rafko_mainframe::RafkoOCLFactory cl_factory =
      (rafko_mainframe::RafkoOCLFactory().select_platform().select_device());
  cl::Context &context = cl_factory.make_context();
  cl::CommandQueue queue(context, cl_factory.selected_device());
  std::unique_ptr<rafko_mainframe::RafkoGPUPhase> test_phase =
      (cl_factory.build<rafko_mainframe::RafkoGPUPhase>(queue, strategy));

  std::vector<double> input_vector(element_count, 666);
  std::vector<double> result_vector(element_count);

  (*test_phase)(input_vector);

  test_phase->load_output(result_vector.data(), element_count);
  REQUIRE_THAT(input_vector,
               Catch::Matchers::Approx(result_vector).margin(0.0000000000001));
}

TEST_CASE("Testing if the simplest OpenCL Implementation is working as "
          "expected with cl::Buffer input",
          "[GPU][Phase][simple]") {
  constexpr const std::size_t element_count = 10;
  std::shared_ptr<EchoStrategy> strategy =
      std::make_unique<EchoStrategy>(element_count);
  rafko_mainframe::RafkoOCLFactory cl_factory =
      (rafko_mainframe::RafkoOCLFactory().select_platform().select_device());
  cl::Context &context = cl_factory.make_context();
  cl::CommandQueue queue(context, cl_factory.selected_device());
  std::unique_ptr<rafko_mainframe::RafkoGPUPhase> test_phase =
      (cl_factory.build<rafko_mainframe::RafkoGPUPhase>(queue, strategy));

  std::vector<double> input_vector(element_count, 666);
  cl::Buffer input_buffer(context, CL_MEM_READ_WRITE,
                          sizeof(double) * element_count);
  std::vector<double> result_vector(element_count);

  cl_int return_value = queue.enqueueWriteBuffer(
      input_buffer, CL_TRUE /* blocking */, 0 /*offset*/,
      sizeof(double) * element_count /*size*/, input_vector.data());
  REQUIRE(return_value == CL_SUCCESS);

  (*test_phase)(input_buffer);

  test_phase->load_output(result_vector.data(), element_count);
  REQUIRE_THAT(input_vector,
               Catch::Matchers::Approx(result_vector).margin(0.0000000000001));
}

TEST_CASE("Testing if the simplest OpenCL Implementation is working as "
          "expected with setting the input cl::Buffer of the phase",
          "[GPU][Phase][simple]") {
  constexpr const std::size_t element_count = 10;
  std::shared_ptr<EchoStrategy> strategy =
      std::make_unique<EchoStrategy>(element_count);
  rafko_mainframe::RafkoOCLFactory cl_factory =
      (rafko_mainframe::RafkoOCLFactory().select_platform().select_device());
  cl::Context &context = cl_factory.make_context();
  cl::CommandQueue queue(context, cl_factory.selected_device());
  std::unique_ptr<rafko_mainframe::RafkoGPUPhase> test_phase =
      (cl_factory.build<rafko_mainframe::RafkoGPUPhase>(queue, strategy));

  std::vector<double> input_vector(element_count, 666);
  std::vector<double> result_vector(element_count);

  cl_int return_value = queue.enqueueWriteBuffer(
      test_phase->get_input_buffer(), CL_TRUE /* blocking */, 0 /*offset*/,
      sizeof(double) * element_count /*size*/, input_vector.data());
  REQUIRE(return_value == CL_SUCCESS);

  (*test_phase)();

  test_phase->load_output(result_vector.data(), element_count);
  REQUIRE_THAT(input_vector,
               Catch::Matchers::Approx(result_vector).margin(0.0000000000001));
}

} /* namespace rafko_mainframe_test */
