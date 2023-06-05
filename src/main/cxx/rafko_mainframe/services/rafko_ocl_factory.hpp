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

#ifndef RAFKO_OCL_FACTORY_H
#define RAFKO_OCL_FACTORY_H

#include "rafko_global.hpp"

#include <CL/opencl.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rafko_gym/models/rafko_agent.hpp"
#include "rafko_gym/models/rafko_dataset.hpp"
#include "rafko_gym/models/rafko_objective.hpp"
#include "rafko_gym/services/rafko_weight_adapter.hpp"
#include "rafko_gym/services/updater_factory.hpp"
#include "rafko_net/services/solution_solver.hpp"

#include "rafko_mainframe/services/rafko_context.hpp"
#include "rafko_mainframe/services/rafko_gpu_phase.hpp"

namespace rafko_mainframe {

class RAFKO_EXPORT RafkoOCLFactory {
public:
  RafkoOCLFactory() {
    cl::Platform::get(&m_platforms);
    RFASSERT_LOG("Number of GPU Platforms: {}", m_platforms.size());
    RFASSERT(0 < m_platforms.size());
  }

  RafkoOCLFactory &select_platform(std::uint32_t platform_index = 0u) {
    RFASSERT_LOG("Selected platform[{}]..", platform_index);
    RFASSERT(platform_index < m_platforms.size());
    RFASSERT_LOG("Platform name: {}",
                 m_platforms[platform_index].getInfo<CL_PLATFORM_NAME>());
    RFASSERT_LOG("Platform version: {}",
                 m_platforms[platform_index].getInfo<CL_PLATFORM_VERSION>());
    RFASSERT_LOG("Platform vendor: {}",
                 m_platforms[platform_index].getInfo<CL_PLATFORM_VENDOR>());
    m_selectedPlatform = platform_index;
    return *this;
  }

  RafkoOCLFactory &select_device(cl_device_type type = CL_DEVICE_TYPE_GPU,
                                 std::uint32_t device_index = 0u) {
    RFASSERT_LOG("Selected device[{}]..", device_index);
    m_platforms[m_selectedPlatform].getDevices(type, &m_devices);
    RFASSERT(device_index < m_devices.size());
    RFASSERT_LOG("Device: {} --> OCL {}",
                 m_devices[device_index].getInfo<CL_DEVICE_NAME>(),
                 m_devices[device_index].getInfo<CL_DEVICE_OPENCL_C_VERSION>());
    m_selectedDevice = device_index;
    return *this;
  }

  cl::Device &selected_device() {
    RFASSERT_LOG("Asking for device[{}]...", m_selectedDevice);
    RFASSERT(m_selectedDevice < m_devices.size());
    return m_devices[m_selectedDevice];
  }

  cl::Context &make_context() {
    RFASSERT_LOG("Creating Context in Factory...");
    RFASSERT(m_selectedDevice < m_devices.size());
    m_context.emplace({m_devices[m_selectedDevice]});
    return m_context.value();
  }

  template <class C, typename... Args>
  std::unique_ptr<C> build(Args &&...args) {
    RFASSERT(0 < m_platforms.size());
    RFASSERT(0 < m_devices.size());
    if (!m_context.has_value())
      make_context();
    return std::make_unique<C>(std::move(m_context.value()),
                               m_devices[m_selectedDevice],
                               std::forward<Args>(args)...);
  }

private:
  std::vector<cl::Platform> m_platforms;
  std::vector<cl::Device> m_devices;
  std::optional<cl::Context> m_context;
  std::uint32_t m_selectedPlatform = -1;
  std::uint32_t m_selectedDevice = -1;
  RFASSERT_SCOPE(RAFKO_GPU_BUILD);
};

} /* namespace rafko_mainframe */

#endif /* RAFKO_OCL_FACTORY_H */
