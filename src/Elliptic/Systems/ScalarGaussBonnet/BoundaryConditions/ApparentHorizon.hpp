// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace sgb::BoundaryConditions {

class ApparentHorizon
    : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "Impose the boundary is a quasi-equilibrium apparent horizon.";

  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The center of the excision surface representing the apparent-horizon "
        "surface";
  };
  struct Rotation {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "The rotational parameters 'Omega' on the surface, which parametrize "
        "the spin of the black hole. The rotational parameters enter the "
        "Dirichlet boundary conditions for the shift in a term "
        "'Omega x (r - Center)', where 'r' are the coordinates on the surface.";
  };
  struct Lapse {
    using type = Options::Auto<
        std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>;
    static constexpr Options::String help =
        "Specify an analytic solution to impose a Dirichlet condition on the "
        "lapse. The analytic solution will be evaluated at coordinates "
        "centered at the apparent horizon. "
        "Alternatively, set this option to 'None' "
        "to impose a zero von-Neumann boundary condition on the lapse. Note "
        "that the latter will not result in the standard Kerr-Schild slicing "
        "for a single black hole.";
  };
  struct NegativeExpansion {
    using type = Options::Auto<
        std::unique_ptr<elliptic::analytic_data::AnalyticSolution>,
        Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Specify an analytic solution to impose its expansion at the excision "
        "surface. The analytic solution will be evaluated at coordinates "
        "centered at the apparent horizon. "
        "If the excision surface lies within the solution's "
        "apparent horizon, the imposed expansion will be negative and thus the "
        "excision surface will lie within an apparent horizon. Alternatively, "
        "set this option to 'None' to impose the expansion is zero at the "
        "excision surface, meaning the excision surface _is_ an apparent "
        "horizon.";
  };

  using options = tmpl::list<Center, Rotation, Lapse, NegativeExpansion>;

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon&) = delete;
  ApparentHorizon& operator=(const ApparentHorizon&) = delete;
  ApparentHorizon(ApparentHorizon&&) = default;
  ApparentHorizon& operator=(ApparentHorizon&&) = default;
  ~ApparentHorizon() = default;

  /// \cond
  explicit ApparentHorizon(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizon);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<ApparentHorizon>(
        center_, rotation_,
        solution_for_lapse_.has_value()
            ? std::make_optional(solution_for_lapse_.value()->get_clone())
            : std::nullopt,
        solution_for_negative_expansion_.has_value()
            ? std::make_optional(
                  solution_for_negative_expansion_.value()->get_clone())
            : std::nullopt);
  }

  ApparentHorizon(
      std::array<double, 3> center, std::array<double, 3> rotation,
      std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
          solution_for_lapse,
      std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
          solution_for_negative_expansion,
      const Options::Context& context = {});

  const std::array<double, 3>& center() const { return center_; }
  const std::array<double, 3>& rotation() const { return rotation_; }
  const std::optional<
      std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>&
  solution_for_lapse() const {
    return solution_for_lapse_;
  }
  const std::optional<
      std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>&
  solution_for_negative_expansion() const {
    return solution_for_negative_expansion_;
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            elliptic::BoundaryConditionType::Neumann,
            // Lapse times conformal factor
            this->solution_for_lapse_.has_value()
                ? elliptic::BoundaryConditionType::Dirichlet
                : elliptic::BoundaryConditionType::Neumann,
            // Shift
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            // Scalar
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::list<
      domain::Tags::FaceNormal<3>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      domain::Tags::UnnormalizedFaceNormalMagnitude<3>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> scalar,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_scalar_gradient,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_scalar,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  using argument_tags_linearized = tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::ConformalFactorMinusOne<DataVector>,
      Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      ::Tags::NormalDotFlux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*> scalar_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*> n_dot_scalar_gradient_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_scalar_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::array<double, 3> center_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, 3> rotation_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
      solution_for_lapse_{};
  std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
      solution_for_negative_expansion_{};
};

}  // namespace sgb::BoundaryConditions
