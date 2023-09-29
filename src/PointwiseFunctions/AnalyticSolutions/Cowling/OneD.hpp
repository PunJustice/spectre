// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <fstream>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling::Solutions {

class OneD : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct ScalarPath {
    using type = std::string;
    static constexpr Options::String help =
        "Path to .txt file containing the solution.";
  };
  struct GridPath {
    using type = std::string;
    static constexpr Options::String help =
        "Path to .txt file containing the grid.";
  };

  struct Resolution {
    using type = size_t;
    static constexpr Options::String help = "Resolution of 1D solution.";
  };

  using options = tmpl::list<ScalarPath, GridPath, Resolution>;
  static constexpr Options::String help{"Loading a solution from my 1D code."};

  OneD() = default;
  OneD(const OneD&) = default;
  OneD& operator=(const OneD&) = default;
  OneD(OneD&&) = default;
  OneD& operator=(OneD&&) = default;
  ~OneD() override = default;
  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<OneD>(scalarpath_, gridpath_, resolution_);
  }
  OneD(std::string scalarpath, std::string gridpath, size_t resolution)
      : scalarpath_(scalarpath), gridpath_(gridpath), resolution_(resolution) {}

  /// \cond
  explicit OneD(CkMigrateMessage* m)
      : elliptic::analytic_data::AnalyticSolution(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OneD);  // NOLINT
  /// \endcond

  template <typename DataType>
  tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
    DataVector radius = magnitude(x).get();
    std::ifstream in;
    in.open(scalarpath_);
    std::vector<double> scalar(resolution_);
    double element;

    if (in.is_open()) {
      size_t i = 0;
      while (in >> element) {
        scalar[i++] = element;
      }
    }
    in.close();

    in.open(gridpath_);
    std::vector<double> grid(resolution_);

    if (in.is_open()) {
      size_t i = 0;
      while (in >> element) {
        grid[i++] = element;
      }
    }
    in.close();

    const intrp::BarycentricRational interpolant(grid, scalar, 3);
    DataVector result = radius;
    size_t i = 0;
    for (double r : radius) {
      result[i++] = interpolant(r);
    };
    return Scalar<DataVector>{result};
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
    p | scalarpath_;
    p | gridpath_;
    p | resolution_;
  }

 private:
  std::string scalarpath_;
  std::string gridpath_;
  size_t resolution_;
};

/// \cond
PUP::able::PUP_ID OneD::my_PUP_ID = 0;  // NOLINT
/// \endcond

bool operator==(const OneD& /*lhs*/, const OneD& /*rhs*/) { return true; }

bool operator!=(const OneD& lhs, const OneD& rhs) { return false; }

}  // namespace Cowling::Solutions
