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
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Printf.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Cowling::Actions {

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
  using fixed_sources_tag = ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi>;

 public:
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& weyl_magnetic_scalar =
        db::get<gr::Tags::WeylMagneticScalar<DataVector>>(box);
    const auto& weyl_electric_scalar =
        db::get<gr::Tags::WeylElectricScalar<DataVector>>(box);

    const auto& previous_solve = db::get<::CurvedScalarWave::Tags::Psi>(box);
    const double epsilon1 = db::get<Cowling::Tags::Epsilon1>(box);
    const double epsilon2 = db::get<Cowling::Tags::Epsilon2>(box);
    const double epsilon4 = db::get<Cowling::Tags::Epsilon4>(box);

    DataVector new_source_dv =
        (epsilon2 * previous_solve.get() / 4. +
         epsilon4 * previous_solve.get() * previous_solve.get() *
             previous_solve.get() / 4.) *
        8. * (weyl_electric_scalar.get() - weyl_magnetic_scalar.get());

    new_source_dv += 8. * epsilon1 *
                     (weyl_electric_scalar.get() - weyl_magnetic_scalar.get());

    // Apply DG mass matrix to the fixed sources if the DG operator is
    // massive
    if (db::get<elliptic::dg::Tags::Massive>(box)) {
      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto& det_inv_jacobian = db::get<
          domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
          box);
      new_source_dv /= get(det_inv_jacobian);
      ::dg::apply_mass_matrix(make_not_null(&new_source_dv), mesh);
    }

    const Scalar<DataVector> new_source{new_source_dv};

    // Increment SolveIteration
    size_t iteration = db::get<Cowling::Tags::SolveIteration>(box) + 1;

    if (is_zeroth_element(element_id, 0)) {
      Parallel::printf(MakeString{}
                       << "Self-Consistent Iteration: " << iteration << "\n");
    }

    db::mutate<fixed_sources_tag, ::Cowling::Tags::SolveIteration>(
        [](const gsl::not_null<Scalar<DataVector>*> field,
           const gsl::not_null<size_t*> solve_iteration,
           const Scalar<DataVector> field_value, const double iteration_value) {
          *field = field_value;
          *solve_iteration = iteration_value;
        },
        make_not_null(&box), new_source, iteration);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Cowling::Actions
