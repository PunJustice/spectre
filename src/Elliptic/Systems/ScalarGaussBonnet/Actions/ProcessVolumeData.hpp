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
#include
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

namespace sgb {
/* Gets all the quantities needed to build SpacetimeQuantitiesComputer
from the databox and the background class.*/
SpacetimeQuantitiesComputer make_computer(db::DataBox<DbTagsList>& box);

namespace Actions {
template <typename BackgroundTag>
struct ProcessVolumeData {
  using flux_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::RolledOffShift<DataVector, 3, Frame::Inertial>,
                 Xcts::Tags::ConformalFactor<DataVector>>;

  using faces_tags = domain::make_faces_tags<3, flux_tags>;

  using imported_and_derived_fields =
      tmpl::list<gr::Tags::WeylElectricScalar<DataVector>,
                 gr::Tags::WeylMagneticScalar<DataVector>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::RolledOffShift<DataVector, 3, Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
                 gr::Tags::SpatialRicci<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;

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
    const auto& spacetime_quantities_computer = make_computer(box);
    const auto& mesh = db::get<domain::Tags::Mesh<3>>(box);
    SpacetimeQuantities spacetime_quantities{mesh.number_of_grid_points()};

    ::Initialization::mutate_assign<imported_and_derived_fields>(
        make_not_null(&box), weyl_electric_scalar, weyl_magnetic_scalar,
        deriv_lapse, deriv_conformal_factor, rolled_off_shift,
        extrinsic_curvature, inv_spatial_metric, spatial_christoffel,
        spatial_ricci);

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

}  // namespace Actions
}  // namespace sgb