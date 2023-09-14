// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Cowling/Geometry.hpp"
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
template <size_t Dim, Geometry BackgroundGeometry>
struct Fluxes;
template <size_t Dim, Geometry BackgroundGeometry>
struct Sources;
}  // namespace Cowling
/// \endcond

namespace Cowling {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Cowling
 * equation on a flat spatial metric in Cartesian coordinates.
 */
template <size_t Dim>
void flat_cartesian_fluxes(
    gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient);

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Cowling equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <size_t Dim>
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_conformal_metric,
                   const tnsr::i<DataVector, Dim>& field_gradient);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Cowling equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
template <size_t Dim>
void add_curved_sources(gsl::not_null<Scalar<DataVector>*> source_for_field,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const tnsr::I<DataVector, Dim>& flux_for_field,
                        const tnsr::i<DataVector, Dim>& deriv_lapse,
                        const Scalar<DataVector>& lapse,
                        const tnsr::i<DataVector, Dim>& conformal_factor_deriv);

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Cowling equation.
 *
 * \see Cowling::FirstOrderSystem
 */
template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field);

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Cowling equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Cowling::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes<Dim, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::i<DataVector, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the curved-space Cowling equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Cowling::FirstOrderSystem
 */
template <size_t Dim>
struct Fluxes<Dim, Geometry::Curved> {
  using argument_tags = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                    const tnsr::II<DataVector, Dim>& inv_conformal_metric,
                    const tnsr::i<DataVector, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
                    const tnsr::II<DataVector, Dim>& inv_conformal_metric,
                    const Scalar<DataVector>& field);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the Cowling equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Cowling::FirstOrderSystem
 */
template <size_t Dim>
struct Sources<Dim, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(gsl::not_null<Scalar<DataVector>*> equation_for_field,
                    const Scalar<DataVector>& field,
                    const tnsr::I<DataVector, Dim>& field_flux);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, Dim>*> equation_for_field_gradient,
      const Scalar<DataVector>& field);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the curved-space Cowling equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Cowling::FirstOrderSystem
 */
template <size_t Dim>
struct Sources<Dim, Geometry::Curved> {
  using argument_tags =
      tmpl::list<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                            Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                               Frame::Inertial>,
                 gr::Tags::Lapse<DataVector>,
                 ::Tags::deriv<Xcts::Tags::ConformalFactor<DataVector>,
                               tmpl::size_t<Dim>, Frame::Inertial>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_field,
      const tnsr::i<DataVector, Dim>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, Dim>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, Dim>& conformal_factor_deriv,
      const Scalar<DataVector>& field,
      const tnsr::I<DataVector, Dim>& field_flux);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, Dim>*> equation_for_field_gradient,
      const tnsr::i<DataVector, Dim>& conformal_christoffel_contracted,
      const tnsr::i<DataVector, Dim>& deriv_lapse,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, Dim>& conformal_factor_deriv,
      const Scalar<DataVector>& field);
};

}  // namespace Cowling
