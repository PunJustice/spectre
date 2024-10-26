// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Equations.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace sgb {

/*!
 * \brief The scalar equation in sGB theories of gravity in the decoupled limit,
 * assuming quasi-stationarity. See \ref sgb for details on the
 * explicit equation.
 */
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using field = ::sgb::Tags::Psi;

 public:
  static constexpr size_t volume_dim = 3;

  using primal_fields =
      tmpl::list<::Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                 ::Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                 ::Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                 field>;
  using primal_fluxes = tmpl::list<
      ::Tags::Flux<::Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                   tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<::Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                   tmpl::size_t<3>, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<field, tmpl::size_t<3>, Frame::Inertial>>;
  // Note that there are many more background fields required for the elliptic
  // solve, however these are numerically imported and so are not listed here.
  using background_fields = tmpl::list<
      // Quantities for Hamiltonian constraint
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ConformalRicciTensor<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ConformalRicciScalar<DataVector>,
      ::Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                   Frame::Inertial>,
      ::Tags::deriv<
          ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
          tmpl::size_t<3>, Frame::Inertial>,
      // Additional quantities for lapse equation
      gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
      // Additional quantities for momentum constraint
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataVector, 3>, 0>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>,
      // Note that this is the plain divergence, i.e. with no
      // Christoffel symbol terms added
      ::Tags::div<
          ::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataVector, 3, Frame::Inertial>>,
      ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ConformalChristoffelFirstKind<DataVector, 3,
                                                  Frame::Inertial>,
      ::Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                   Frame::Inertial>>;
  using inv_metric_tag =
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>;

  using fluxes_computer = Fluxes;
  using fluxes_computer_linearized = LinearizedFluxes;
  using sources_computer = Sources;
  using sources_computer_linearized = LinearizedSources;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<3>;
};

}  // namespace sgb
