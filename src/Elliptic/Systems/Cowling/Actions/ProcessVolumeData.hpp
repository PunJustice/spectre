// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Actions {

/*!
 * \brief Wait for data from a volume data file to arrive and directly move it
 * into the DataBox
 *
 * Monitors `importers::Tags::VolumeData` in the element's inbox and moves the
 * received data directly into the `FieldTagsList` in the DataBox.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <typename FieldTagsList>
struct ProcessVolumeData {
  using inbox_tags = tmpl::list<importers::Tags::VolumeData<FieldTagsList>>;

  using flux_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::RolledOffShift<DataVector, 3, Frame::Inertial>,
                 Xcts::Tags::ConformalFactor<DataVector>>;

  using faces_tags = domain::make_faces_tags<3, flux_tags>;

  using imported_and_derived_fields = tmpl::list<
      gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>, gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::Shift<DataVector, 3>, Xcts::Tags::ConformalFactor<DataVector>,
      ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::RolledOffShift<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3>,
      Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;

  using simple_tags =
      tmpl::flatten<tmpl::list<imported_and_derived_fields, faces_tags>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox =
        tuples::get<importers::Tags::VolumeData<FieldTagsList>>(inboxes);
    // Using `0` for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    const auto received_data = inbox.find(0_st);
    if (received_data == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto& element_data = received_data->second;

    const auto& coords =
        db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);
    DataVector r = magnitude(coords).get();

    const auto& full_shift =
        tuples::get<gr::Tags::Shift<DataVector, 3>>(element_data);
    const auto& shift_excess =
        tuples::get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
            element_data);
    tnsr::I<DataVector, 3> shift;
    DataVector shift_background;

    const double rolloff_location =
        db::get<Cowling::Tags::RolloffLocation>(box);

    const double rolloff_rate = db::get<Cowling::Tags::RolloffRate>(box);

    for (size_t i = 0; i < 3; i++) {
      shift_background = full_shift.get(i) - shift_excess.get(i);
      shift.get(i) = shift_excess.get(i) +
                     (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                         shift_background / 2;
    }

    const auto& lapse = tuples::get<gr::Tags::Lapse<DataVector>>(element_data);
    const auto spatial_metric =
        tuples::get<gr::Tags::SpatialMetric<DataVector, 3>>(element_data);
    const auto& extrinsic_curvature =
        tuples::get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(element_data);
    const auto& conformal_factor =
        tuples::get<Xcts::Tags::ConformalFactor<DataVector>>(element_data);
    const auto& inv_conformal_metric = tuples::get<
        Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
        element_data);
    const auto& mesh = db::get<domain::Tags::Mesh<3>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(box);

    const auto& deriv_lapse = partial_derivative(lapse, mesh, inv_jacobian);
    const auto& deriv_conformal_factor =
        partial_derivative(conformal_factor, mesh, inv_jacobian);

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
        gr::christoffel_second_kind(deriv_spatial_metric, inv_spatial_metric);

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

    ::Initialization::mutate_assign<imported_and_derived_fields>(
        make_not_null(&box), weyl_electric_scalar, weyl_magnetic_scalar, lapse,
        deriv_lapse, full_shift, conformal_factor, deriv_conformal_factor,
        shift, spatial_metric, extrinsic_curvature, shift_excess);

    DirectionMap<3, Scalar<DataVector>> sliced_lapse;
    DirectionMap<3, tnsr::I<DataVector, 3, Frame::Inertial>> sliced_shift;
    DirectionMap<3, Scalar<DataVector>> sliced_conformal_factor;

    for (const auto& direction : Direction<3>::all_directions()) {
      data_on_slice(make_not_null(&(sliced_lapse)[direction]), lapse,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
      data_on_slice(make_not_null(&(sliced_shift)[direction]), shift,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
      data_on_slice(make_not_null(&(sliced_conformal_factor)[direction]),
                    conformal_factor, mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    }

    ::Initialization::mutate_assign<faces_tags>(make_not_null(&box),
                                                sliced_lapse, sliced_shift,
                                                sliced_conformal_factor);

    inbox.erase(received_data);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Cowling::Actions
