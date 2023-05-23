// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for the Poisson system

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"

/// \cond
class DataVector;
/// \endcond

/*!
 * \ingroup EllipticSystemsGroup
 * \brief Items related to solving a Poisson equation \f$-\Delta u(x)=f(x)\f$.
 */
namespace Poisson {

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
struct Epsilon {
  static std::string name() { return "Epsilon"; }
  using type = double;
  static constexpr Options::String help{
      "Epsilon for self-consistent iterations."};
};

}  // namespace OptionTags
namespace Tags {

/*!
 * \brief The scalar field \f$u(x)\f$ to solve for
 */
struct Field : db::SimpleTag {
  using type = Scalar<DataVector>;
};

/*!
 * \brief The number of self consistent iterations done.
 */
struct SolveIteration : db::SimpleTag {
  using type = size_t;
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
 * \brief Epsilon for self-consistent iterations.
 */
struct Epsilon : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Epsilon>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double epsilon) { return epsilon; }
};

}  // namespace Tags

}  // namespace Poisson
