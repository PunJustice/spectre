// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Tags.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace sgb {
struct Fluxes;
struct Sources;
struct LinearizedSources;
}  // namespace sgb
/// \endcond

namespace sgb {

/*!
 * \brief Compute the fluxes $F^i=\left(\psi^{-4} \tilde{\gamma}^{ij}
 * -\alpha^{2} \beta^i \beta^j \right) \partial_j \Psi(x)$ for the scalar
 * equation in sGB gravity on a conformal metric $\tilde{\gamma}_{ij}$.
 */
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& field_gradient);

/*!
 * \brief Compute the fluxes $F^i=\left(\psi^{-4} \tilde{\gamma}^{ij}
 * -\alpha^{2} \beta^i \beta^j \right) \partial_j \Psi(x)$ for the scalar
 * equation in sGB gravity on a conformal metric $\tilde{\gamma}_{ij}$ on a face
 * normal.
 */
void face_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_conformal_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse,
                 const Scalar<DataVector>& conformal_factor,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field);

/*!
 * \brief Adds the source terms arising from the $\Box \Psi$ term in the
 * equation of motion for the scalar field: $S=-\tilde{\Gamma}^i_{ij}F^j-F^j
 * \alpha^{-1} \partial_j \alpha - 6F^j \psi^{-1} \partial_j \psi$.*/
void add_curved_sources(gsl::not_null<Scalar<DataVector>*> source_for_field,
                        const tnsr::i<DataVector, 3>& christoffel_contracted,
                        const tnsr::I<DataVector, 3>& flux_for_field,
                        const tnsr::i<DataVector, 3>& deriv_lapse,
                        const Scalar<DataVector>& lapse,
                        const Scalar<DataVector>& conformal_factor,
                        const tnsr::i<DataVector, 3>& conformal_factor_deriv);

/*!
 * \brief Add the sGB coupling term $\mathcal{R} f'(\Psi)=2(E-B)(\epsilon_2 \Psi
 * + \epsilon_4 \Psi^3)$.
 */
void add_GB_terms(gsl::not_null<Scalar<DataVector>*> scalar_tensor_equation,
                  double eps2, double eps4,
                  const Scalar<DataVector>& weyl_electric,
                  const Scalar<DataVector>& weyl_magnetic,
                  const Scalar<DataVector>& field);

/*!
 * \brief Add sources arising from linearising the sGB coupling term.
 */
void add_linearized_GB_terms(
    gsl::not_null<Scalar<DataVector>*> linearized_scalar_tensor_equation,
    double eps2, double eps4, const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the scalar equation in sGB gravity on
 * a spatial metric \f$\gamma_{ij}\f$.
 */
struct Fluxes {
  using argument_tags = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                 Frame::Inertial>,
      Tags::RolloffLocation, Tags::RolloffRate>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags =
      tmpl::list<Tags::RolloffLocation, Tags::RolloffRate>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
      gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
      const double& rolloff_location, const double& rolloff_rate,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& shift_excess,
      const Scalar<DataVector>& scalar,
      const tnsr::i<DataVector, 3>& conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::i<DataVector, 3>& scalar_gradient);
  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          flux_for_lapse_times_conformal_factor,
      gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
      gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::I<DataVector, 3>& face_normal_vector,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& shift_excess,
      const Scalar<DataVector>& scalar);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the scalar equation in sGB gravity on a
 * spatial metric \f$\gamma_{ij}\f$.
 */
struct Sources {
  using argument_tags = tmpl::list<
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataVector, 3>, 0>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Xcts::Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>,
      ::Tags::div<
          ::Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
              DataVector, 3, Frame::Inertial>>,
      ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::ConformalChristoffelFirstKind<DataVector, 3,
                                                  Frame::Inertial>,
      ::Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                   Frame::Inertial>,
      ::Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                   Frame::Inertial>,
      ::Xcts::Tags::ConformalRicciScalar<DataVector>, Tags::Epsilon2,
      Tags::Epsilon4, gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
      gsl::not_null<Scalar<DataVector>*> lapse_equation,
      gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
      gsl::not_null<Scalar<DataVector>*> scalar_equation,
      const Scalar<DataVector>& conformal_energy_density,
      const Scalar<DataVector>& conformal_stress_trace,
      const tnsr::I<DataVector, 3>& conformal_momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
          div_longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar, const double& eps2,
      const double& eps4, const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& shift_excess,
      const Scalar<DataVector>& scalar,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
      const tnsr::I<DataVector, 3>& scalar_flux);
};

/*!
 * \brief Add the linearised sources \f$S_A\f$ for the scalar equation in sGB
 * gravity on a spatial metric \f$\gamma_{ij}\f$.
 */
struct LinearizedSources {
  using argument_tags = tmpl::push_back<
      typename Sources::argument_tags,
      ::Xcts::Tags::ConformalFactorMinusOne<DataVector>,
      ::Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      ::Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      ::Tags::Flux<::Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                   tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Flux<::Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                   tmpl::size_t<3>, Frame::Inertial>,
      ::Xcts::Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
      gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
      gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
      gsl::not_null<Scalar<DataVector>*> linearized_scalar_equation,
      const Scalar<DataVector>& conformal_energy_density,
      const Scalar<DataVector>& conformal_stress_trace,
      const tnsr::I<DataVector, 3>& conformal_momentum_density,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const Scalar<DataVector>& dt_extrinsic_curvature_trace,
      const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>&
          longitudinal_shift_background_minus_dt_conformal_metric,
      const tnsr::I<DataVector, 3>&
      /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
      const tnsr::ii<DataVector, 3>& conformal_metric,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const Scalar<DataVector>& conformal_ricci_scalar, const double& eps2,
      const double& eps4, const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& shift_excess,
      const tnsr::I<DataVector, 3>& conformal_factor_flux,
      const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
      const Scalar<DataVector>& conformal_factor_correction,
      const Scalar<DataVector>& lapse_times_conformal_factor_correction,
      const tnsr::I<DataVector, 3>& shift_excess_correction,
      const Scalar<DataVector>& scalar_correction,
      const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
      const tnsr::I<DataVector, 3>&
          lapse_times_conformal_factor_flux_correction,
      const tnsr::II<DataVector, 3>& longitudinal_shift_excess_correction,
      const tnsr::I<DataVector, 3>& scalar_flux_correction);
};

}  // namespace sgb
