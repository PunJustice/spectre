// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
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
namespace Cowling {
struct Fluxes;
struct Sources;
}  // namespace Cowling
/// \endcond

namespace Cowling {

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Cowling equation on a spatial metric \f$\gamma_{ij}\f$.
 */
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const tnsr::i<DataVector, 3>& field_gradient);

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Cowling equation on a spatial metric \f$\gamma_{ij}\f$
 * on a face normal.
 */
void face_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_conformal_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Cowling equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
void add_curved_sources(gsl::not_null<Scalar<DataVector>*> source_for_field,
                        const tnsr::i<DataVector, 3>& christoffel_contracted,
                        const tnsr::I<DataVector, 3>& flux_for_field,
                        const tnsr::i<DataVector, 3>& deriv_lapse,
                        const Scalar<DataVector>& lapse);

/*!
 * \brief Add the linearized sources for the puncture equation.
 *
 * Adds $-\frac{d}{du}(\beta \left(\alpha \left(1 + u\right) + 1\right)^{-7})$.
 *
 * \see Punctures
 */
void add_GB_terms(gsl::not_null<Scalar<DataVector>*> cowling_equation,
                  const double eps2, const double eps4,
                  const Scalar<DataVector>& weyl_electric,
                  const Scalar<DataVector>& weyl_magnetic,
                  const Scalar<DataVector>& field);

void add_linearized_GB_terms(
    gsl::not_null<Scalar<DataVector>*> linearized_cowling_equation,
    const double eps2, const double eps4,
    const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction);

struct Fluxes {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
                 gr::Tags::ShiftPlusVelocity<DataVector, 3, Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = false;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
                    const Scalar<DataVector>& field,
                    const tnsr::i<DataVector, 3>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
                    const tnsr::i<DataVector, 3>& face_normal,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the curved-space Cowling equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Cowling::FirstOrderSystem
 */
struct Sources {
  using argument_tags =
      tmpl::list<gr::Tags::SpatialChristoffelSecondKindContracted<
                     DataVector, 3, Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>, Tags::Epsilon2, Tags::Epsilon4,
                 gr::Tags::WeylElectricScalar<DataVector>,
                 gr::Tags::WeylMagneticScalar<DataVector>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_field,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse, const double& eps2, const double& eps4,
      const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
      const tnsr::I<DataVector, 3>& field_flux);
};

struct LinearizedSources {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3,
                                                       Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      gr::Tags::Lapse<DataVector>, ::CurvedScalarWave::Tags::Psi,
      Tags::Epsilon2, Tags::Epsilon4, gr::Tags::WeylElectricScalar<DataVector>,
      gr::Tags::WeylMagneticScalar<DataVector>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> linearized_equation_for_field,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse, const Scalar<DataVector>& field,
      const double& eps2, const double& eps4,
      const Scalar<DataVector>& weyl_electric,
      const Scalar<DataVector>& weyl_magnetic,
      const Scalar<DataVector>& field_correction,
      const tnsr::I<DataVector, 3>& field_flux_correction);
};

}  // namespace Cowling
