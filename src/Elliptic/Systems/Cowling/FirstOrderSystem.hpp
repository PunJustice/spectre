// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Cowling/Equations.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling {

/*!
 * \brief The puncture equation, formulated as a set of coupled first-order
 * partial differential equations
 *
 * See \ref Punctures for details on the puncture equation. Since it is just a
 * flat-space Poisson equation with nonlinear sources, we can reuse the
 * Euclidean Poisson fluxes.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using field = ::CurvedScalarWave::Tags::Psi;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields = tmpl::list<field>;
  using primal_fluxes =
      tmpl::list<::Tags::Flux<field, tmpl::size_t<3>, Frame::Inertial>>;

  using background_fields = tmpl::list<
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3,
                                                       Frame::Inertial>,
      gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>, gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
      gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
      gr::Tags::ShiftPlusVelocity<DataVector, 3, Frame::Inertial>>;
  using inv_metric_tag =
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>;

  using fluxes_computer = Fluxes;
  using sources_computer = Sources;
  using sources_computer_linearized = LinearizedSources;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};

}  // namespace Cowling
