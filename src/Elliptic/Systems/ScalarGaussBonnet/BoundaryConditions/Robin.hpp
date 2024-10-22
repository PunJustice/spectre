// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace sgb::BoundaryConditions {

class Robin : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Impose Robin boundary conditions at the outer boundary. "
      "They incur an error of order 1/R^2, where R is the outer radius.";

  Robin() = default;
  Robin(const Robin&) = default;
  Robin& operator=(const Robin&) = default;
  Robin(Robin&&) = default;
  Robin& operator=(Robin&&) = default;
  ~Robin() override = default;

  /// \cond
  explicit Robin(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Robin);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override;

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override;

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3, Frame::Inertial>>;
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
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3, Frame::Inertial>>;
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
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;
};

bool operator==(const Robin& lhs, const Robin& rhs);

bool operator!=(const Robin& lhs, const Robin& rhs);

}  // namespace sgb::BoundaryConditions
