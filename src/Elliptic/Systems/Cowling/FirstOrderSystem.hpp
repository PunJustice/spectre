// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Cowling::FirstOrderSystem

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Cowling/Equations.hpp"
#include "Elliptic/Systems/Cowling/Geometry.hpp"
#include "Elliptic/Systems/Cowling/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling {

/*!
 * \brief The Cowling equation formulated as a set of coupled first-order PDEs.
 *
 * \details This system formulates the sGB equation \f$-\Delta_\gamma u(x) =
 * \epsilon f(\phi) R_{GB}\f$ on a background metric \f$\gamma_{ij}\f$ as the
 * set of coupled first-order PDEs
 *
 * \f[
 * -\partial_i \gamma^{ij} v_j(x) - \Gamma^i_{ij}\gamma^{jk}v_k = \epsilon
 * f(\phi) R_{GB} \\
 * -\partial_i u(x) + v_i(x) = 0
 * \f]
 *
 * where we have chosen the field gradient as an auxiliary variable \f$v_i\f$
 * and where \f$\Gamma^i_{jk}=\frac{1}{2}\gamma^{il}\left(\partial_j\gamma_{kl}
 * +\partial_k\gamma_{jl}-\partial_l\gamma_{jk}\right)\f$ are the Christoffel
 * symbols of the second kind of the background metric \f$\gamma_{ij}\f$. The
 * background metric \f$\gamma_{ij}\f$ and the Christoffel symbols derived from
 * it are assumed to be independent of the variables \f$u\f$ and \f$v_i\f$, i.e.
 * constant throughout an iterative elliptic solve. We also will treat terms
 * proportional to \f$\epsilon\f$ as fixed, where we will solve for them through
 * a self-consistent iteration scheme.
 *
 * The system can be formulated in terms of these fluxes and sources (see
 * `elliptic::protocols::FirstOrderSystem`):
 *
 * \f{align*}
 * F^i_u &= \gamma^{ij} v_j(x) \\
 * S_u &= -\Gamma^i_{ij}\gamma^{jk}v_k \\
 * f_u &= \epsilonf(\phi) R_{GB}  \\
 * F^i_{v_j} &= u \delta^i_j \\
 * S_{v_j} &= v_j \\
 * f_{v_j} &= 0 \text{.}
 * \f}
 *
 */
template <size_t Dim>
struct FirstOrderSystem
    : tt::ConformsTo<elliptic::protocols::FirstOrderSystem> {
 private:
  using field = ::CurvedScalarWave::Tags::Psi;
  using field_gradient =
      ::Tags::deriv<field, tmpl::size_t<Dim>, Frame::Inertial>;

 public:
  static constexpr size_t volume_dim = Dim;

  using primal_fields = tmpl::list<field>;
  using auxiliary_fields = tmpl::list<field_gradient>;

  // We just use the standard `Flux` prefix because the fluxes don't have
  // symmetries and we don't need to give them a particular meaning.
  using primal_fluxes =
      tmpl::list<::Tags::Flux<field, tmpl::size_t<Dim>, Frame::Inertial>>;
  using auxiliary_fluxes = tmpl::list<
      ::Tags::Flux<field_gradient, tmpl::size_t<Dim>, Frame::Inertial>>;

  using background_fields = tmpl::list<
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                 Frame::Inertial>>;
  using inv_metric_tag =
      Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>;

  using fluxes_computer = Fluxes<Dim, Cowling::Geometry::Curved>;
  using sources_computer = Sources<Dim, Cowling::Geometry::Curved>;

  using boundary_conditions_base =
      elliptic::BoundaryConditions::BoundaryCondition<Dim>;
};
}  // namespace Cowling
