// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
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
 * \brief Processes loaded data from Xcts solve.
 */

struct ProcessData {
 private:
 public:
  using simple_tags = tmpl::list<gr::Tags::WeylElectricScalar<DataVector>,
                                 gr::Tags::WeylMagneticScalar<DataVector>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& lapse = db::get<gr::Tags::Lapse<DataVector>>(box);
    const auto& spatial_metric =
        db::get<gr::Tags::SpatialMetric<DataVector, 3>>(box);
    const auto& extrinsic_curvature =
        db::get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(box);
    const auto& conformal_factor =
        db::get<Xcts::Tags::ConformalFactor<DataVector>>(box);
    const auto& inv_conformal_metric = db::get<
        Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
        box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);

    const auto det_spatial_metric = determinant(spatial_metric);
    const Scalar<DataVector> sqrt_det_spatial_metric{
        sqrt(get(det_spatial_metric))};

    tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric;
    tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_spatial_metric),
                                  inv_conformal_metric(ti::I, ti::J) /
                                      conformal_factor() / conformal_factor() /
                                      conformal_factor() / conformal_factor());

    const auto deriv_spatial_metric =
        partial_derivative(spatial_metric, mesh, inv_jacobian);

    const auto spatial_christoffel =
        christoffel_second_kind(deriv_spatial_metric, inv_spatial_metric);

    const auto deriv_spatial_christoffel =
        partial_derivative(spatial_christoffel, mesh, inv_jacobian);

    const auto spatial_ricci =
        gr::ricci_tensor(spatial_christoffel, deriv_spatial_christoffel);

    const auto deriv_extrinsic_curvature =
        partial_derivative(extrinsic_curvature, mesh, inv_jacobian);
    tnsr::ijj<DataVector, 3, Frame::Inertial> cov_deriv_extrinsic_curvature{};
    tenex::evaluate<ti::i, ti::j, ti::k>(
        make_not_null(&cov_deriv_extrinsic_curvature),
        deriv_extrinsic_curvature(ti::i, ti::j, ti::k) -
            spatial_christoffel(ti::L, ti::i, ti::j) *
                extrinsic_curvature(ti::l, ti::k) -
            spatial_christoffel(ti::L, ti::i, ti::k) *
                extrinsic_curvature(ti::j, ti::l));

    const auto weyl_electric = gr::weyl_electric(
        spatial_ricci, extrinsic_curvature, inv_spatial_metric);
    const auto weyl_electric_scalar =
        gr::weyl_electric_scalar(weyl_electric, inv_spatial_metric);
    const auto weyl_magnetic = gr::weyl_magnetic(
        cov_deriv_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric);
    const auto weyl_magnetic_scalar =
        gr::weyl_magnetic_scalar(weyl_magnetic, inv_spatial_metric);

    ::Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), weyl_electric_scalar, weyl_magnetic_scalar);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Cowling::Actions
