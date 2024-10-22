// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/ScalarGaussBonnet/BoundaryConditions/Robin.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Robin.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace sgb::BoundaryConditions {

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Robin::get_clone() const {
  return std::make_unique<Robin>(*this);
}

std::vector<elliptic::BoundaryConditionType> Robin::boundary_condition_types()
    const {
  return {6, elliptic::BoundaryConditionType::Neumann};
}

void Robin::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> scalar,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_scalar_gradient,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_scalar*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  const auto r = magnitude(x);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_conformal_factor_gradient, *conformal_factor_minus_one, r);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_lapse_times_conformal_factor_gradient,
      *lapse_times_conformal_factor_minus_one, r);
  ::Xcts::BoundaryConditions::robin_boundary_condition_shift(
      n_dot_longitudinal_shift_excess, *shift_excess, deriv_shift_excess, r,
      face_normal);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_scalar_gradient, *scalar, r);
}

template <>
void Robin<Xcts::Equations::HamiltonianLapseAndShift>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*> scalar_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*> n_dot_scalar_gradient_correction,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
    const tnsr::i<DataVector, 3>& /*deriv_scalar_correction*/,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  const auto r = magnitude(x);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_conformal_factor_gradient_correction, *conformal_factor_correction,
      r);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_lapse_times_conformal_factor_gradient_correction,
      *lapse_times_conformal_factor_correction, r);
  ::Xcts::BoundaryConditions::robin_boundary_condition_shift(
      n_dot_longitudinal_shift_excess_correction, *shift_excess_correction,
      deriv_shift_excess_correction, r, face_normal);
  ::Xcts::BoundaryConditions::robin_boundary_condition_scalar(
      n_dot_scalar_gradient_correction, *scalar_correction, r);
}

bool operator==(const Robin& /*lhs*/, const Robin& /*rhs*/) { return true; }

bool operator!=(const Robin& lhs, const Robin& rhs) { return not(lhs == rhs); }

PUP::able::PUP_ID Robin::my_PUP_ID = 0;  // NOLINT

}  // namespace sgb::BoundaryConditions
