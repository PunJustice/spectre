// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Solutions {

/// The trivial solution \f$u=0\f$ of a Poisson equation. Useful as initial
/// guess.
template <size_t Dim>
class Inverser : public elliptic::analytic_data::AnalyticSolution {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{"1/r profile for initial guess."};

  Inverser() = default;
  Inverser(const Inverser&) = default;
  Inverser& operator=(const Inverser&) = default;
  Inverser(Inverser&&) = default;
  Inverser& operator=(Inverser&&) = default;
  ~Inverser() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Inverser>(*this);
  }

  /// \cond
  explicit Inverser(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Inverser);  // NOLINT
  /// \endcond

  template <typename DataType>
  tuples::TaggedTuple<Tags::Field> variables(
      const tnsr::I<DataType, Dim>& x, tmpl::list<Tags::Field> /*meta*/) const {
    DataVector r = magnitude(x).get();
    DataVector result = 1 / r;

    return Scalar<DataVector>{result};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags = tmpl::list<
        Tags::Field,
        ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
        ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>,
        ::Tags::FixedSource<Tags::Field>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID Inverser<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
bool operator==(const Inverser<Dim>& /*lhs*/, const Inverser<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const Inverser<Dim>& lhs, const Inverser<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Cowling::Solutions
