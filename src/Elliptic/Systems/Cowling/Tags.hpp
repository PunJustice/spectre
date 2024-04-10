// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Cowling system

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving a Cowling equation \f$-\Delta u(x)=f(x)\f$.
 */
namespace Cowling {

namespace OptionTags {

/*!
 * \brief The maximum number of self-consistent iterations to be done.
 */
struct MaxIterations {
  static std::string name() { return "MaxIterations"; }
  using type = size_t;
  static constexpr Options::String help{
      "The maximum number of self-consistent iterations to be done."};
};

/*!
 * \brief Epsilon for self-consistent iterations.
 */
struct Epsilon1 {
  static std::string name() { return "Epsilon1"; }
  using type = double;
  static constexpr Options::String help{
      "Epsilon1 for self-consistent iterations."};
};

struct Epsilon2 {
  static std::string name() { return "Epsilon2"; }
  using type = double;
  static constexpr Options::String help{
      "Epsilon2 for self-consistent iterations."};
};

struct Epsilon4 {
  static std::string name() { return "Epsilon4"; }
  using type = double;
  static constexpr Options::String help{
      "Epsilon4 for self-consistent iterations."};
};
struct DampingParameter : db::SimpleTag {
  static std::string name() { return "DampingParameter"; }
  using type = double;
  static constexpr Options::String help{
      "Damping parameter for self-consistent iterations."};
};

}  // namespace OptionTags
namespace Tags {

/*!
 * \brief The number of self consistent iterations done.
 */
struct SolveIteration : db::SimpleTag {
  using type = size_t;
};

/*!
 * \brief The previous solve, for damping in self-consistent solves.
 */
struct PreviousSolve : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief Damping parameter for self-consistent solve.
 */
struct DampingParameter : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::DampingParameter>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double damping_parameter) {
    return damping_parameter;
  }
};

/*!
 * \brief The maximum number of self-consistent iterations to be done.
 */
struct MaxIterations : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::MaxIterations>;
  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t max_iterations) {
    return max_iterations;
  }
};

/*!
 * \brief Epsilon1 for self-consistent iterations.
 */
struct Epsilon1 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon1>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

/*!
 * \brief Epsilon2 for self-consistent iterations.
 */
struct Epsilon2 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon2>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

/*!
 * \brief Epsilon4 for self-consistent iterations.
 */
struct Epsilon4 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon4>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

struct Flux : db::SimpleTag {
  using type = tnsr::II<DataVector, 3, Frame::Inertial>;
};
struct MoveDerivToPhi : ::CurvedScalarWave::Tags::Phi<3>, db::ComputeTag {
 public:
  using base = ::CurvedScalarWave::Tags::Phi<3>;
  using return_type = typename base::type;
  static void function(gsl::not_null<return_type*> result,
                       const tnsr::i<DataVector, 3, Frame::Inertial>& deriv) {
    *result = deriv;
  }

  using argument_tags =
      tmpl::list<::Tags::deriv<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>>;
};

struct UpdatePi : ::CurvedScalarWave::Tags::Pi, db::ComputeTag {
 public:
  using base = ::CurvedScalarWave::Tags::Pi;
  using return_type = typename base::type;
  static void function(gsl::not_null<return_type*> result,
                       const tnsr::i<DataVector, 3, Frame::Inertial>& deriv,
                       const tnsr::I<DataVector, 3>& shift,
                       const Scalar<DataVector>& lapse) {
    result->get() = get(dot_product(shift, deriv)) / get(lapse);
  }
  using argument_tags =
      tmpl::list<::Tags::deriv<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>,
                 gr::Tags::Shift<DataVector, 3>, gr::Tags::Lapse<DataVector>>;
};

// struct ComputeFlux : Cowling::Tags::Flux, db::ComputeTag {
//  public:
//   using base = ::Cowling::Tags::Flux;
//   using return_type = typename base::type;
//   static void function(
//       gsl::not_null<return_type*> result,
//       const tnsr::II<DataVector, 3, Frame::Inertial>&
//       inverse_conformal_metric, const tnsr::I<DataVector, 3>& shift, const
//       Scalar<DataVector>& lapse const Scalar<DataVector>&
//           conformal_factor) {
//     result->get() = get(dot_product(shift, deriv)) / get(lapse) / get(lapse);
//   }
//   using argument_tags = tmpl::list<
//       Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
//       gr::Tags::Shift<DataVector, 3>, gr::Tags::Lapse<DataVector>,
//       Xcts::Tags::ConformalFactor<DataVector>>;
// };
}  // namespace Tags

}  // namespace Cowling
