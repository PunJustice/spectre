// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/ScalarGaussBonnet/BoundaryConditions/ApparentHorizon.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizon.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace sgb::BoundaryConditions {

ApparentHorizon::ApparentHorizon(
    std::array<double, 3> center, std::array<double, 3> rotation,
    std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
        solution_for_lapse,
    std::optional<std::unique_ptr<elliptic::analytic_data::AnalyticSolution>>
        solution_for_negative_expansion,
    const Options::Context& /*context*/)
    : center_(center),
      rotation_(rotation),
      // NOLINTNEXTLINE(performance-move-const-arg)
      solution_for_lapse_(std::move(solution_for_lapse)),
      solution_for_negative_expansion_(
          // NOLINTNEXTLINE(performance-move-const-arg)
          std::move(solution_for_negative_expansion)) {}

void ApparentHorizon::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> /*scalar*/,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const gsl::not_null<Scalar<DataVector>*> /*n_dot_scalar_gradient*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& /*deriv_scalar*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  ::Xcts::BoundaryConditions::apparent_horizon_impl<::Xcts::Geometry::Curved>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, center_, rotation_, solution_for_lapse_,
      solution_for_negative_expansion_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

void ApparentHorizon::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*> /*scalar_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const gsl::not_null<
        Scalar<DataVector>*> /*n_dot_scalar_gradient_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_scalar_correction*/,
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
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  ::Xcts::BoundaryConditions::linearized_apparent_horizon_impl<
      ::Xcts::Geometry::Curved>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, center_, solution_for_lapse_,
      solution_for_negative_expansion_, face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, x,
      extrinsic_curvature_trace, longitudinal_shift_background,
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      n_dot_longitudinal_shift_excess, inv_conformal_metric,
      conformal_christoffel_second_kind);
}

void ApparentHorizon::pup(PUP::er& p) {
  p | center_;
  p | rotation_;
  p | solution_for_lapse_;
  p | solution_for_negative_expansion_;
}

PUP::able::PUP_ID ApparentHorizon::my_PUP_ID = 0;  // NOLINT

}  // namespace sgb::BoundaryConditions
