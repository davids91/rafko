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

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>
#if (RAFKO_USES_OPENCL)
#include <string>
#endif /*(RAFKO_USES_OPENCL)*/

#include "rafko_gym/models/rafko_backpropagation_data.hpp"
#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"

namespace rafko_gym {

/**
 * @brief A class representing an operation inside the backpropagation logic of
 * reverse mode autodiff. each operation is collected with the help of the
 * components of a Neuron ( input-, transfer- and spike function ) and objective
 * in each of the classes that inherit from this class; While this class
 * provides a common base which aims to eliminate the stack restrictions present
 * in recursive solutions, by storing every operation in a vector, and providing
 * the chance to upload the operation dependencies into the vector when
 * prompted.
 */
class RAFKO_EXPORT RafkoBackpropagationOperation {
public:
  using Dependency = std::shared_ptr<RafkoBackpropagationOperation>;
  using DependencyParameter =
      std::pair<Autodiff_operations, std::vector<std::uint32_t>>;
  using DependencyParameters = std::vector<DependencyParameter>;
  using DependencyRegister = std::function<void(
      std::vector<std::shared_ptr<RafkoBackpropagationOperation>>)>;
  using DependencyRequest =
      std::optional<std::pair<DependencyParameters, DependencyRegister>>;

  RafkoBackpropagationOperation(RafkoBackpropagationData &data,
                                const rafko_net::RafkoNet &network,
                                std::uint32_t operation_index,
                                Autodiff_operations type);

  virtual ~RafkoBackpropagationOperation() = default;

  /**
   * @brief   Provides a list of dependencies this operation requires, along
   * with a register function to handle when the requested dependencies are
   * actually uploaded
   *
   * @return  A list of Dependency parameters this operation needs along with a
   * function to call when the dependencies are construted
   */
  virtual DependencyRequest request_dependencies() = 0;

  /**
   * @brief     Calculates the forward propagation value for this operation
   *
   * @param[in]     network_input     Const access to the provided network input
   * array
   */
  virtual void calculate_value(const std::vector<double> &network_input) = 0;

  /**
   * @brief     Calculates the backward propagation value ( derivative) for this
   * operation based on the given parameters
   *
   * @param[in]     d_w_index         The index of the weights for the basis of
   * the derivative
   * @param[in]     network_input     Const access to the provided network input
   * array
   * @param[in]     label_data        Const access to the provided labels array
   */
  virtual void calculate_derivative(std::uint32_t d_w_index,
                                    const std::vector<double> &network_input,
                                    const std::vector<double> &label_data) = 0;

#if (RAFKO_USES_OPENCL)
  /**
   * @brief   Provides the needed local variables for this operation in OpenCL
   * kernel code
   *
   * @return  The local variables required for the kernel code generated by the
   * object
   */
  virtual std::string local_declaration_operation() const = 0;

  /**
   * @brief   Tells the requested size of the local group for the generated kernel code 
   *
   * @return  The number of threads the generated kernel code uses
   */
  virtual std::uint32_t optimal_thread_count() const { return 1u; }

  /**
   * @brief     Generates GPU kernel enumerations
   *
   * @return    An enumerator to be ised in the GPU kernel
   */
  static std::string get_kernel_enums() {
    return R"(
      typedef enum autodiff_operations_e{
        ad_operation_unknown = 0,
        ad_operation_objective_d,
        ad_operation_neuron_spike_d,
        ad_operation_neuron_transfer_d,
        ad_operation_neuron_input_d,
        ad_operation_neuron_bias_d,
      }autodiff_operations_t __attribute__ ((aligned));
    )";
  }

#endif /*(RAFKO_USES_OPENCL)*/

  /**
   * @brief   Provides the derivative value for the given past and weight index
   *
   * @brief   past_index    The previous run the network was executed in, where
   * the derivative was generated
   * @brief   d_w_index     The index of the weight the derivative is based in
   *
   * @return    The derivative value of the current operation
   */
  double get_derivative(std::uint32_t past_index,
                        std::uint32_t d_w_index) const;
  /**
   * @brief   Provides the forward propagated value for the given past index
   *
   * @brief   past_index    The previous run the network was executed in, where
   * the value was generated
   *
   * @return    The value of the current propagation
   */
  double get_value(std::uint32_t past_index) const;

  /**
   * @brief     Returns with whether or not the required dependencies of the
   * operation are registered
   *
   * @return    true, if all required dependencies are registered
   */
  bool constexpr are_dependencies_registered() const {
    return m_dependenciesRegistered;
  }

  /**
   * @brief     Returns with whether or not the stored value is already
   * calculated through forward propagation
   *
   * @return    true, if the value of the operation is calculated
   */
  bool constexpr is_value_processed() const { return m_valueProcessed; }

  /**
   * @brief     Returns with whether or not the operation is calculated fully
   *
   * @return    true, if both the value and the derivative values are already
   * calculated
   */
  bool constexpr is_processed() const {
    return (m_valueProcessed && m_derivativeProcessed);
  }

  /**
   * @brief     Get the actual operation index for this operation.
   *
   * @return    The index stored in the operation
   */
  virtual std::uint32_t get_operation_index() const { return m_operationIndex; }

  /**
   * @brief     Provides the type of the operation
   *
   * @return    enum value of type Autodiff_operations
   */
  constexpr Autodiff_operations get_type() const { return m_type; }

  /**
   * @brief     Provides whether or not the operation index provides correct
   * value. In most operations, the index can be determined at object
   * initialization, however, some special operations might need to be placed
   * later on to the array, because some other operations might depend on them
   * too (the order of dependency must hold true in the array --> no item can
   * depend on items before it). In these cases the operation index might be set
   * dynamically later.
   *
   * @return    True if the operation index returns its final value placed in
   * the operations aray
   */
  virtual bool operation_index_finalised() { return true; }

  /**
   * @brief   Provides dependencies used by calculation
   *
   * @return  constant access to the Operation dependencies
   */
  virtual std::vector<Dependency> get_own_dependencies() = 0;

  /**
   * @brief     Returns with the maximum of the index values of its dependencies
   *
   * @return    Index value of the maximum dependencies
   */
  std::uint32_t get_max_dependency_index();

  /**
   * @brief     Inserts the given dependency reference to be kept by this
   * operation
   *
   * @param     dep     An operation this operation depends on
   */
  void insert_dependency(Dependency dep) { m_addedDependencies.push_back(dep); }

  /**
   * @brief     Provides a vector of the stored dependency references
   *
   * @return    A vector of the stored dependency references
   */
  std::vector<Dependency> get_dependencies() {
    std::vector<Dependency> deps(m_addedDependencies);
    std::vector<Dependency> own_deps(get_own_dependencies());
    deps.insert(deps.end(), own_deps.begin(), own_deps.end());
    return deps;
  }

protected:
  RafkoBackpropagationData &m_data;
  const rafko_net::RafkoNet &m_network;
  const std::uint32_t m_operationIndex;

  /**
   * @brief     Sets internal procesed flag to false for both value and
   * derivative of this operation
   */
  void constexpr reset_processed() {
    m_valueProcessed = false;
    m_derivativeProcessed = false;
  }

  /**
   * @brief     Sets internal procesed flag to false for forward propagated
   * value of this operation
   */
  void constexpr set_value_processed() { m_valueProcessed = true; }

  /**
   * @brief     Sets internal procesed flag to false for derivative of this
   * operation
   */
  void constexpr set_derivative_processed() { m_derivativeProcessed = true; }

  /**
   * @brief     Sets internal procesed flag to true for both value and
   * derivative of this operation
   */
  void constexpr set_processed() {
    m_valueProcessed = true;
    m_derivativeProcessed = true;
  }

  /**
   * @brief     Sets dependencies registered flag to true in this operation; The
   * dependencies of the operation are already registered, when this flag is set
   * to true.
   */
  void constexpr set_registered() { m_dependenciesRegistered = true; }

  /**
   * @brief   Sets the current derivative value for the given weight index of
   * this operation
   *
   * @brief   d_w_index     The index of the weight the derivative is based in
   * @brief   value         The value to set
   */
  void set_derivative(std::uint32_t d_w_index, double value);

  /**
   * @brief   Sets the current forward propagated value of this operation
   *
   * @brief   value         The value to set
   */
  void set_value(double value);

private:
  const Autodiff_operations m_type;
  bool m_valueProcessed = false;
  bool m_derivativeProcessed = false;
  bool m_dependenciesRegistered = false;
  std::vector<Dependency> m_addedDependencies;
};

} /* namespace rafko_gym */

#endif /* RAFKO_BACKPROPAGATION_OPERATION_H */
