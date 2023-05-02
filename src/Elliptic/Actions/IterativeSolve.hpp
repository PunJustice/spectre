// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace elliptic::Actions {

/*!
 * \brief Inialitises flux for next iterative solve.
 *
 * This action increments the SolveIteration tag, tracking how many iterations
 * have occured. It then mutates the FixedSource, updating it with the solution
 * from the previuos solve. Must be used in conjuction with CheckConvergence.
 *
 * Uses:
 * - DataBox:
 *   - `primal_fields`
 *   - `Poisson::Tags::SolveIteration`
 *   - `Tags::AnalyticSource<Dim, Frame::Inertial>`
 *   - `domain::Tags::Mesh<Dim>`
 *   - `domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
 *      Frame::Inertial>`
 *
 * DataBox:
 * - Mutates:
 *   - `::Tags::FixedSource<Poisson::Tags::Field>`
 *   - `Poisson::Tags::SolveIteration`
 */
struct IterativeSolve {
 private:
  using fixed_sources_tag = ::Tags::FixedSource<::Poisson::Tags::Field>;

  using analytic_sources_tag = ::Tags::AnalyticSource<::Poisson::Tags::Field>;

 public:
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& analytic_source = db::get<analytic_sources_tag>(box);
    const auto& previous_solve = db::get<Poisson::Tags::Field>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<3>>(box);
    const auto& inverse_jacobian =
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(box);
    const auto first_derivative =
        partial_derivative(previous_solve, mesh, inverse_jacobian);
    const auto second_derivative =
        partial_derivative(first_derivative, mesh, inverse_jacobian);

    DataVector new_source_dv = analytic_source[0] +
                                  0.001 * second_derivative.get(0, 1);

    // Apply DG mass matrix to the fixed sources if the DG operator is
    // massive
    if (db::get<elliptic::dg::Tags::Massive>(box)) {
      const auto& det_inv_jacobian = db::get<
          domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
          box);
      new_source_dv /= get(det_inv_jacobian);
      ::dg::apply_mass_matrix(make_not_null(&new_source_dv), mesh);
    }

    const Scalar<DataVector>new_source{new_source_dv};

    // Increment SolveIteration
    size_t iteration = db::get<Poisson::Tags::SolveIteration>(box) + 1;
    Parallel::printf(std::to_string(iteration));
    db::mutate<fixed_sources_tag, ::Poisson::Tags::SolveIteration>(
        make_not_null(&box),
        [](const gsl::not_null<Scalar<DataVector>*> field,
           const gsl::not_null<size_t*> solve_iteration,
           const Scalar<DataVector> field_value, const double iteration_value) {
          *field = field_value;
          *solve_iteration = iteration_value;
        },
        new_source, iteration);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace elliptic::Actions
