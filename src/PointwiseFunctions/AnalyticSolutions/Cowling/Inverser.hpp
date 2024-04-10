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
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Solutions {

template <size_t Dim>
class Inverser : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = "Scaled Amplitude.";
  };

  using options = tmpl::list<Amplitude>;
  static constexpr Options::String help{"A/r profile for initial guess."};

  Inverser() = default;
  Inverser(const Inverser&) = default;
  Inverser& operator=(const Inverser&) = default;
  Inverser(Inverser&&) = default;
  Inverser& operator=(Inverser&&) = default;
  ~Inverser() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Inverser>(amplitude_);
  }

  Inverser(double amplitude) : amplitude_(amplitude) {}

  /// \cond
  explicit Inverser(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Inverser);  // NOLINT
  /// \endcond

  template <typename DataType>
  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
    DataVector r = magnitude(x).get();
    DataVector result = amplitude_ / r;
    return Scalar<DataVector>{result};
  }

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<::CurvedScalarWave::Tags::Psi,
                                    tmpl::size_t<3>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x,
      tmpl::list<::Tags::deriv<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>> /*meta*/) const {
    DataVector r = magnitude(x).get();
    DataVector dx = -amplitude_ * get<0>(x) / (r * r * r);
    DataVector dy = -amplitude_ * get<1>(x) / (r * r * r);
    DataVector dz = -amplitude_ * get<2>(x) / (r * r * r);
    return tnsr::i<DataType, Dim>{{{dx, dy, dz}}};
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
    p | amplitude_;
  }

 private:
  double amplitude_;
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
