// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
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
                        const Scalar<DataVector>& lapse,
                        const tnsr::i<DataVector, 3>& conformal_factor_deriv);

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Cowling equation.
 *
 * \see Cowling::FirstOrderSystem
 */
void auxiliary_fluxes(gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_gradient,
                      const Scalar<DataVector>& field);

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the curved-space Cowling equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Cowling::FirstOrderSystem
 */

struct Fluxes {
  using argument_tags = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::Shift<DataVector, 3>,
      gr::Tags::Lapse<DataVector>>;
  using volume_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
                    const tnsr::i<DataVector, 3>& field_gradient);
  static void apply(gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_gradient,
                    const tnsr::II<DataVector, 3>& inv_conformal_metric,
                    const tnsr::I<DataVector, 3>& shift,
                    const Scalar<DataVector>& lapse,
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
      tmpl::list<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                            Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>,
                 ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<3>, Frame::Inertial>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_field,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3>& conformal_factor_deriv,
      const Scalar<DataVector>& field,
      const tnsr::I<DataVector, 3>& field_flux);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, 3>*> equation_for_field_gradient,
      const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, 3>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3>& conformal_factor_deriv,
      const Scalar<DataVector>& field);
};

}  // namespace Cowling
