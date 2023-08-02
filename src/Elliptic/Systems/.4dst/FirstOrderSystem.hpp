// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/4dst/FluxesAndSources.hpp"
#include "Elliptic/Systems/4dst/Geometry.hpp"
#include "Elliptic/Systems/4dst/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace 4dst {

  /*!
   * \brief The Extended Conformal Thin Sandwich (XCTS) decomposition for 4DST
   * theories, in the form written by Kovacs. This is essentially the \ref Xcts
   * equations, with an additional Poisson equation for the scalar field
   * \f$\phi\f$. For now, we focus on the minimally coupled limit.
   */
  template <Equations EnabledEquations, Geometry ConformalGeometry,
            int ConformalMatterScale>
  struct FirstOrderSystem
      : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
   public:
    static constexpr Equations enabled_equations = EnabledEquations;
    static constexpr Geometry conformal_geometry = ConformalGeometry;
    static constexpr int conformal_matter_scale = ConformalMatterScale;

   private:
    using conformal_factor = Tags::ConformalFactor<DataVector>;
    using conformal_factor_gradient =
        ::Tags::deriv<conformal_factor, tmpl::size_t<3>, Frame::Inertial>;
    using lapse_times_conformal_factor =
        Tags::LapseTimesConformalFactor<DataVector>;
    using lapse_times_conformal_factor_gradient =
        ::Tags::deriv<lapse_times_conformal_factor, tmpl::size_t<3>,
                      Frame::Inertial>;
    using shift_excess = Tags::ShiftExcess<DataVector, 3, Frame::Inertial>;
    using shift_strain = Tags::ShiftStrain<DataVector, 3, Frame::Inertial>;
    using longitudinal_shift_excess =
        Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>;
    using gb_scalar = Tags::GBScalar<DataVector>;
    using gb_scalar_gradient =
        ::Tags::deriv<gb_scalar, tmpl::size_t<3>, Frame::Inertial>;

   public:
    static constexpr size_t volume_dim = 3;

    using primal_fields = tmpl::flatten<tmpl::list<
        conformal_factor,
        tmpl::conditional_t<
            EnabledEquations == Equations::HamiltonianAndLapse or
                EnabledEquations == Equations::HamiltonianLapseAndShift,
            lapse_times_conformal_factor, tmpl::list<>>,
        tmpl::conditional_t<EnabledEquations ==
                                Equations::HamiltonianLapseAndShift,
                            shift_excess, tmpl::list<>>,
        gb_scalar>>;
    using auxiliary_fields = tmpl::flatten<tmpl::list<
        conformal_factor_gradient,
        tmpl::conditional_t<
            EnabledEquations == Equations::HamiltonianAndLapse or
                EnabledEquations == Equations::HamiltonianLapseAndShift,
            lapse_times_conformal_factor_gradient, tmpl::list<>>,
        tmpl::conditional_t<EnabledEquations ==
                                Equations::HamiltonianLapseAndShift,
                            shift_strain, tmpl::list<>>,
        gb_scalar_gradient>>;

    // As fluxes we use the gradients with raised indices for the Hamiltonian
    // and lapse equation, and the longitudinal shift excess for the momentum
    // constraint. The gradient fluxes don't have symmetries and no particular
    // meaning so we use the standard `Flux` tags, but for the symmetric
    // longitudinal shift we use the corresponding symmetric tag.
    using primal_fluxes = tmpl::flatten<tmpl::list<
        ::Tags::Flux<conformal_factor, tmpl::size_t<3>, Frame::Inertial>,
        tmpl::conditional_t<
            EnabledEquations == Equations::HamiltonianAndLapse or
                EnabledEquations == Equations::HamiltonianLapseAndShift,
            ::Tags::Flux<lapse_times_conformal_factor, tmpl::size_t<3>,
                         Frame::Inertial>,
            tmpl::list<>>,
        tmpl::conditional_t<EnabledEquations ==
                                Equations::HamiltonianLapseAndShift,
                            longitudinal_shift_excess, tmpl::list<>>>>;
    using auxiliary_fluxes = db::wrap_tags_in<::Tags::Flux, auxiliary_fields,
                                              tmpl::size_t<3>, Frame::Inertial>;

    using background_fields = tmpl::flatten<tmpl::list<
        // Quantities for Hamiltonian constraint
        gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>,
                            ConformalMatterScale>,
        gr::Tags::TraceExtrinsicCurvature<DataVector>,
        tmpl::conditional_t<
            ConformalGeometry == Geometry::Curved,
            tmpl::list<
                Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
                Tags::ConformalRicciScalar<DataVector>,
                Tags::ConformalChristoffelContracted<DataVector, 3,
                                                     Frame::Inertial>,
                ::Tags::deriv<
                    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>>,
            tmpl::list<>>,
        tmpl::conditional_t<
            EnabledEquations ==
                Equations::Hamiltonian,
            Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
                DataVector>,
            tmpl::list<>>,
        // Additional quantities for lapse equation
        tmpl::conditional_t<
            EnabledEquations == Equations::HamiltonianAndLapse or
                EnabledEquations == Equations::HamiltonianLapseAndShift,
            tmpl::list<
                gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>,
                                    ConformalMatterScale>,
                ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>,
            tmpl::list<>>,
        tmpl::conditional_t<
            EnabledEquations ==
                Equations::HamiltonianAndLapse,
            tmpl::list<
                Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
                Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>>,
            tmpl::list<>>,
        // Additional quantities for momentum constraint
        tmpl::conditional_t<
            EnabledEquations ==
                Equations::HamiltonianLapseAndShift,
            tmpl::list<
                gr::Tags::Conformal<
                    gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>,
                    ConformalMatterScale>,
                ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                              tmpl::size_t<3>, Frame::Inertial>,
                Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
                Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                    DataVector, 3, Frame::Inertial>,
                // Note that this is the plain divergence, i.e. with no
                // Christoffel symbol terms added
                ::Tags::div<
                    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                        DataVector, 3, Frame::Inertial>>,
                tmpl::conditional_t<
                    ConformalGeometry == Geometry::Curved,
                    tmpl::list<
                        Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                        Tags::ConformalChristoffelFirstKind<DataVector, 3,
                                                            Frame::Inertial>,
                        Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                             Frame::Inertial>>,
                    tmpl::list<>>>,
            tmpl::list<>>>>;
    using inv_metric_tag = tmpl::conditional_t<
        ConformalGeometry == Geometry::FlatCartesian, void,
        Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;

    using fluxes_computer = Fluxes<EnabledEquations, ConformalGeometry>;
    using sources_computer =
        Sources<EnabledEquations, ConformalGeometry, ConformalMatterScale>;
    using sources_computer_linearized =
        LinearizedSources<EnabledEquations, ConformalGeometry,
                          ConformalMatterScale>;

    using boundary_conditions_base =
        elliptic::BoundaryConditions::BoundaryCondition<3>;
  };

}  // namespace 4dst
