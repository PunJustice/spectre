// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
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
 * \brief Initialize the "fixed sources" of the elliptic equations, i.e. their
 * variable-independent source term \f$f(x)\f$
 *
 * This action initializes \f$f(x)\f$ in an elliptic system of PDEs \f$-div(F) +
 * S = f(x)\f$.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `db::wrap_tags_in<::Tags::FixedSource, primal_fields>`
 */
template <typename System, typename BackgroundTag>
struct InitializeFixedSources {
 private:
  using fixed_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>>;

  using analytic_sources_tag = ::Tags::Variables<
      db::wrap_tags_in<::Tags::AnalyticSource, typename System::primal_fields>>;

 public:
  using const_global_cache_tags =
      tmpl::list<elliptic::dg::Tags::Massive, BackgroundTag>;
  using simple_tags = tmpl::list<fixed_sources_tag, analytic_sources_tag,
                                 Poisson::Tags::SolveIteration>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& background = db::get<BackgroundTag>(box);

    // Stores the analytic source, that doesn't change between iterations.
    // Setting our initial guess to be zero, this is equal to the fixed source
    // for the first solve. This will be updated each iteration with a different
    // action.
    auto fixed_sources = elliptic::util::get_analytic_data<
        typename fixed_sources_tag::tags_list>(background, box,
                                               inertial_coords);

    // For an initial guess of 0, the first fixed_source is simply the analytic
    // source.
    auto analytic_sources = fixed_sources;

    // Apply DG mass matrix to the fixed sources if the DG operator is
    // massive
    if (db::get<elliptic::dg::Tags::Massive>(box)) {
      const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
      const auto& det_inv_jacobian = db::get<
          domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
          box);
      fixed_sources /= get(det_inv_jacobian);
      ::dg::apply_mass_matrix(make_not_null(&fixed_sources), mesh);
    }

    // Here we set the number of iterative solves done so far to 0.
    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(fixed_sources),
        std::move(analytic_sources), 0_st);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace elliptic::Actions
