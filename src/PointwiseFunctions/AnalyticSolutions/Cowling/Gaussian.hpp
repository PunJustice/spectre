// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Solutions {

class Gaussian : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct Amplitude {
    using type = double;
    static constexpr Options::String help = "Amplitude of Gaussian.";
  };
  struct Width {
    using type = double;
    static constexpr Options::String help = "Width of Gaussian.";
  };
  struct Center {
    using type = double;
    static constexpr Options::String help = "Center of Gaussian.";
  };

  using options = tmpl::list<Amplitude, Width, Center>;
  static constexpr Options::String help{
      "Gaussian initial guess for scalar-profile solves."};

  Gaussian() = default;
  Gaussian(const Gaussian&) = default;
  Gaussian& operator=(const Gaussian&) = default;
  Gaussian(Gaussian&&) = default;
  Gaussian& operator=(Gaussian&&) = default;
  ~Gaussian() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<Gaussian>(amplitude_, width_, center_);
  }
  Gaussian(double amplitude, double width, double center)
      : amplitude_(amplitude), width_(width), center_(center) {}

  /// \cond
  explicit Gaussian(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Gaussian);  // NOLINT
  /// \endcond

  template <typename DataType>
  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
    DataVector r = magnitude(x).get();
    DataVector result = amplitude_ * exp(-pow(r - center_, 2) / width_);
    return Scalar<DataVector>{result};
  }

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<::CurvedScalarWave::Tags::Psi,
                                    tmpl::size_t<3>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::Tags::deriv<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                               Frame::Inertial>> /*meta*/) const {
    DataVector r = magnitude(x).get();
    DataVector dx = -amplitude_ * (2 * (r - center_) / width_) * get<0>(x) *
                    exp(-pow(r - center_, 2) / width_) / r;
    DataVector dy = -amplitude_ * (2 * (r - center_) / width_) * get<1>(x) *
                    exp(-pow(r - center_, 2) / width_) / r;
    DataVector dz = -amplitude_ * (2 * (r - center_) / width_) * get<2>(x) *
                    exp(-pow(r - center_, 2) / width_) / r;
    return tnsr::i<DataType, 3>{{{dx, dy, dz}}};
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    using supported_tags =
        tmpl::list<::CurvedScalarWave::Tags::Psi,
                   ::Tags::deriv<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                                 Frame::Inertial>,
                   ::Tags::Flux<::CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                                Frame::Inertial>,
                   ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi>>;
    static_assert(tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                                   supported_tags>>::value == 0,
                  "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }
  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | amplitude_;
    p | width_;
    p | center_;
  }

 private:
  double amplitude_;
  double width_;
  double center_;
};

/// \cond
PUP::able::PUP_ID Gaussian::my_PUP_ID = 0;  // NOLINT
/// \endcond

bool operator==(const Gaussian& /*lhs*/, const Gaussian& /*rhs*/) {
  return true;
}

bool operator!=(const Gaussian& lhs, const Gaussian& rhs) {
  return not(lhs == rhs);
}

}  // namespace Cowling::Solutions
