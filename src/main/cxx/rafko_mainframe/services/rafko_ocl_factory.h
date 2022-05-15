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

#include "rafko_global.h"

#include <memory>
#include <mutex>
#include <vector>
#include <CL/opencl.hpp>

#include "rafko_net/services/solution_solver.h"
#include "rafko_gym/models/rafko_environment.h"
#include "rafko_gym/models/rafko_objective.h"
#include "rafko_gym/models/rafko_agent.h"
#include "rafko_gym/services/rafko_weight_adapter.h"
#include "rafko_gym/services/updater_factory.h"

#include "rafko_mainframe/services/rafko_gpu_phase.h"
#include "rafko_mainframe/services/rafko_context.h"

namespace rafko_mainframe {

class RAFKO_FULL_EXPORT RafkoOCLFactory{
public:
  RafkoOCLFactory(){
    cl::Platform::get(&platforms);
    RFASSERT_LOG("Number of GPU Platforms: {}", platforms.size());
    RFASSERT( 0 < platforms.size() );
  }

  RafkoOCLFactory& select_platform(std::uint32_t platform_index = 0u){
    RFASSERT_LOG("Selected platform[{}]..", platform_index);
    RFASSERT( platform_index < platforms.size() );
    RFASSERT_LOG("Platform name: {}", platforms[selected_platform].getInfo<CL_PLATFORM_NAME>());
    RFASSERT_LOG("Platform version: {}", platforms[selected_platform].getInfo<CL_PLATFORM_VERSION>());
    RFASSERT_LOG("Platform vendor: {}", platforms[selected_platform].getInfo<CL_PLATFORM_VENDOR>());
    selected_platform = platform_index;
    return *this;
  }

  RafkoOCLFactory& select_device(cl_device_type type = CL_DEVICE_TYPE_GPU, std::uint32_t device_index = 0u){
    RFASSERT_LOG("Selected device[{}]..", device_index);
    platforms[selected_platform].getDevices(type, &devices);
    RFASSERT( device_index < devices.size() );
    RFASSERT_LOG(
      "Device: {} --> OCL {}",
      devices[device_index].getInfo<CL_DEVICE_NAME>(),
      devices[device_index].getInfo<CL_DEVICE_OPENCL_C_VERSION>()
    );
    selected_device = device_index;
    return *this;
  }

  template<class C, typename... Args>
  std::unique_ptr<C> build(Args && ... args){
    RFASSERT( 0 < platforms.size() );
    RFASSERT( 0 < devices.size() );
    cl::Context context({devices[selected_device]});
    return std::make_unique<C>(
      std::move(context), devices[selected_device],
      std::forward<Args>(args)...
    );
  }

private:
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  std::uint32_t selected_platform = 0u;
  std::uint32_t selected_device = 0u;
  RFASSERT_SCOPE(RAFKO_GPU_BUILD);
};


} /* namespace rafko_mainframe */

#endif /* RAFKO_OCL_FACTORY_H */
