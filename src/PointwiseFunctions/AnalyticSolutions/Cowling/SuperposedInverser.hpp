// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Solutions {

template <size_t Dim>
class SuperposedInverser : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct AmplitudeA {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude A.";
  };
  struct AmplitudeB {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude B.";
  };
  struct LocationA {
    using type = double;
    static constexpr Options::String help = "Location of black hole A.";
  };
  struct LocationB {
    using type = double;
    static constexpr Options::String help = "location of black hole B.";
  };

  using options = tmpl::list<AmplitudeA, AmplitudeB, LocationA, LocationB>;
  static constexpr Options::String help{
      "Superposed 1/r initial guesses for SBVP problem."};

  SuperposedInverser() = default;
  SuperposedInverser(const SuperposedInverser&) = default;
  SuperposedInverser& operator=(const SuperposedInverser&) = default;
  SuperposedInverser(SuperposedInverser&&) = default;
  SuperposedInverser& operator=(SuperposedInverser&&) = default;
  ~SuperposedInverser() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<SuperposedInverser>(amplitudea_, amplitudeb_,
                                                locationa_, locationb_);
  }

  SuperposedInverser(double amplitudea, double amplitudeb, double locationa,
                     double locationb)
      : amplitudea_(amplitudea),
        amplitudeb_(amplitudeb),
        locationa_(locationa),
        locationb_(locationb) {}

  /// \cond
  explicit SuperposedInverser(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SuperposedInverser);  // NOLINT
  /// \endcond

  template <typename DataType>
  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
    DataVector rminusra =
        sqrt((get<0>(x) - locationa_) * (get<0>(x) - locationa_) +
             get<1>(x) * get<1>(x) + get<2>(x) * get<2>(x));
    DataVector rminusrb =
        sqrt((get<0>(x) - locationb_) * (get<0>(x) - locationb_) +
             get<1>(x) * get<1>(x) + get<2>(x) * get<2>(x));
    DataVector result = amplitudea_ / rminusra + amplitudeb_ / rminusrb;
    return Scalar<DataVector>{result};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags =
        tmpl::list<::CurvedScalarWave::Tags::Psi,
                   ::Tags::deriv<::CurvedScalarWave::Tags::Psi,
                                 tmpl::size_t<Dim>, Frame::Inertial>,
                   ::Tags::Flux<::CurvedScalarWave::Tags::Psi,
                                tmpl::size_t<Dim>, Frame::Inertial>,
                   ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | amplitudea_;
    p | amplitudeb_;
    p | locationa_;
    p | locationb_;
  }

 private:
  double amplitudea_;
  double amplitudeb_;
  double locationa_;
  double locationb_;
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID SuperposedInverser<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <size_t Dim>
bool operator==(const SuperposedInverser<Dim>& /*lhs*/,
                const SuperposedInverser<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const SuperposedInverser<Dim>& lhs,
                const SuperposedInverser<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Cowling::Solutions
