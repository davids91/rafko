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

#ifndef RAFKO_BACKPROPAGATION_OPERATION_H
#define RAFKO_BACKPROPAGATION_OPERATION_H

#include "rafko_global.hpp"

#include <memory>
#include <vector>
#include <optional>
#include <algorithm>
#if(RAFKO_USES_OPENCL)
#include <string>
#endif/*(RAFKO_USES_OPENCL)*/

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_gym/models/rafko_backpropagation_data.hpp"

namespace rafko_gym{

/**
 * @brief A class representing an operation inside the backpropagation logic of reverse mode autodiff.
 * each operation is collected with the help of the components of a Neuron ( input-, transfer- and spike function )
 * and objective in each of the classes that inherit from this class; While this class provides a common base
 * which aims to eliminate the stack restrictions present in recursive solutions, by storing every operation
 * in a vector, and providing the chance to upload the operation dependencies into the vector when prompted.
 */
class RafkoBackpropagationOperation;
class RAFKO_EXPORT RafkoBackpropagationOperation{
public:
  using Dependency = std::shared_ptr<RafkoBackpropagationOperation>;
  using DependencyParameter = std::pair<Autodiff_operations,std::vector<std::uint32_t>>;
  using DependencyParameters = std::vector<DependencyParameter>;
  using DependencyRegister = std::function<void(std::vector<std::shared_ptr<RafkoBackpropagationOperation>>)>;
  using DependencyRequest = std::optional<std::pair<DependencyParameters,DependencyRegister>>;

  RafkoBackpropagationOperation(
    RafkoBackpropagationData& data, const rafko_net::RafkoNet& network,
    std::uint32_t operation_index, Autodiff_operations type
  )
  : m_data(data)
  , m_network(network)
  , m_operationIndex(operation_index)
  , m_type(type)
  {
  }

  virtual ~RafkoBackpropagationOperation() = default;
  virtual DependencyRequest upload_dependencies_to_operations() = 0;

  virtual void calculate_value(const std::vector<double>& network_input) = 0;
  virtual void calculate_derivative(
    std::uint32_t d_w_index, const std::vector<double>& network_input, const std::vector<double>& label_data
  ) = 0;

  #if(RAFKO_USES_OPENCL)
  virtual std::string local_declaration_operation() const = 0;
  virtual std::string value_kernel_operation(
    std::string network_input_array, std::string weight_array,
    std::string operations_value_array, std::string operations_array_size
  ) const = 0;
  virtual std::string derivative_kernel_operation(
    std::string network_input_array, std::string label_array, std::string weight_array,
    std::string operations_value_array, std::string operations_derivative_array,
    std::string operations_array_size, std::string d_operations_array_size
  ) const = 0;
  virtual bool is_multi_worker() const{
    return false;
  }
  #endif/*(RAFKO_USES_OPENCL)*/


  double get_derivative(std::uint32_t past_index, std::uint32_t d_w_index) const{
    return m_data.get_derivative(past_index, get_operation_index(), d_w_index);
  }

  double get_value(std::uint32_t past_index) const{
    return m_data.get_value(past_index, get_operation_index());
  }

  bool constexpr are_dependencies_registered() const{
    return m_dependenciesRegistered;
  }

  bool constexpr is_value_processed() const{
    return m_valueProcessed;
  }

  bool constexpr is_processed() const{
    return (m_valueProcessed && m_derivativeProcessed);
  }

  virtual std::uint32_t get_operation_index() const{
    return m_operationIndex;
  }

  constexpr Autodiff_operations get_type() const{
    return m_type;
  }


  virtual bool operation_index_finalised(){
    return true; /*!Note: Descendants might want to have operation index set dynamically */
  }

  std::uint32_t get_max_dependency_index();

  void insert_dependency(Dependency dep){
    m_addedDependencies.push_back(dep);
  }

  std::vector<Dependency> get_dependencies(){
    std::vector<Dependency> deps(m_addedDependencies);
    std::vector<Dependency> own_deps(get_own_dependencies());
    deps.insert(deps.begin(), own_deps.begin(), own_deps.end());
    return deps;
  }

protected:
  RafkoBackpropagationData& m_data;
  const rafko_net::RafkoNet& m_network;
  const std::uint32_t m_operationIndex;

  virtual std::vector<Dependency> get_own_dependencies() = 0;

  void constexpr reset_processed(){
    m_valueProcessed = false;
    m_derivativeProcessed = false;
  }

  void constexpr set_value_processed(){
    m_valueProcessed = true;
  }

  void constexpr set_derivative_processed(){
    m_derivativeProcessed = true;
  }

  void constexpr set_processed(){
    m_valueProcessed = true;
    m_derivativeProcessed = true;
  }

  void constexpr set_registered(){
    m_dependenciesRegistered = true;
  }

  void set_derivative(std::uint32_t d_w_index, double value){
    m_data.set_derivative(get_operation_index(), d_w_index, value);
  }

  void set_value(double value){
    m_data.set_value(get_operation_index(), value);
  }

private:
  const Autodiff_operations m_type;
  bool m_valueProcessed = false;
  bool m_derivativeProcessed = false;
  bool m_dependenciesRegistered = false;
  std::vector<Dependency> m_addedDependencies;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_OPERATION_H */
