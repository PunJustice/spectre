// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/ScalarGaussBonnet/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace sgb {

void curved_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& shift_rolloff,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::i<DataVector, 3>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_conformal_metric);
  const DataVector conformal_factor_power = pow<4>(1 + get(conformal_factor));
  const DataVector shift_term =
      get(dot_product(shift_rolloff, field_gradient)) /
      square(1 + get(lapse_times_conformal_factor_minus_one) /
                     get(conformal_factor));

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= conformal_factor_power;
    flux_for_field->get(i) -= shift_rolloff.get(i) * shift_term;
  }
}

void face_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& shift_rolloff,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::i<DataVector, 3>& face_normal,
    const Scalar<DataVector>& field) {
  raise_or_lower_index(flux_for_field, face_normal, inv_conformal_metric);
  const DataVector conformal_factor_power = pow<4>(1 + get(conformal_factor));
  const DataVector shift_term = get(dot_product(shift_rolloff, face_normal)) /
                                square(1 + get(lapse) / get(conformal_factor));

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= conformal_factor_power;
    flux_for_field->get(i) -= shift_rolloff.get(i) * shift_term;
    flux_for_field->get(i) *= get(field);
  }
}

void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& conformal_factor_flux) {
  tnsr::i<DataVector, 3> lapse_times_conformal_factor_deriv =
      raise_or_lower_index(lapse_times_conformal_factor_flux, conformal_metric);
  get(*source_for_field) -=
      get(dot_product(lapse_times_conformal_factor_deriv, flux_for_field)) /
      (1 + get(lapse_times_conformal_factor_minus_one));
  get(*source_for_field) -=
      get(dot_product(conformal_christoffel_contracted, flux_for_field));
  tnsr::i<DataVector, 3> conformal_factor_deriv =
      raise_or_lower_index(conformal_factor_flux, conformal_metric);
  get(*source_for_field) -=
      5. * get(dot_product(conformal_factor_deriv, flux_for_field)) /
      (1 + get(conformal_factor_minus_one));
}

tnsr::I<DataVector, 3> curved_sources(
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& conformal_factor_flux) {
  tnsr::I<DataVector, 3> result;
  const auto raised_christoffel =
      raise_or_lower_index(conformal_christoffel_contracted, conformal_metric);
  for (size_t i = 0; i < 3; i++) {
    result.get(i) = lapse_times_conformal_factor_flux.get(i) /
                    (1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) += raised_christoffel.get(i);
    result.get(i) += 5. * conformal_factor_flux.get(i) /
                     (1 + get(conformal_factor_minus_one));
  }
}

// Returns linearisation of source w/ respect to xcts variables
tnsr::I<DataVector, 3> source_part_linearization(
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) {
  tnsr::I<DataVector, 3> result;
  for (size_t i = 0; i < 3; i++) {
    result.get(i) = -get(lapse_times_conformal_factor_correction) *
                    lapse_times_conformal_factor_flux.get(i) /
                    square(1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) += lapse_times_conformal_factor_flux.get(i) /
                     (1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) -= 5. * conformal_factor_correction *
                     conformal_factor_flux.get(i) /
                     square(1 + get(conformal_factor_minus_one));
    result.get(i) += 5. * conformal_factor_flux_correction.get(i) /
                     (1 + get(conformal_factor_minus_one))
  }
  return result;
}

// Returns linearisation of flux w/ respect to xcts variables
tnsr::I<DataVector, 3> flux_part_linearization(
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::i<DataVector, 3>& scalar,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_rolloff,
    const tnsr::I<DataVector, 3>& shift_excess_correction) {
  tnsr::I<DataVector, 3> result;
  const auto scalar_deriv = partial_derivative(scalar, mesh, inv_jacobian);
  const auto raise_scalar_deriv =
      raise_or_lower_index(scalar_deriv, inv_conformal_metric);
  for (size_t i = 0; i < 3; i++) {
    result.get(i) = -4 * get(conformal_factor_correction) *
                    raise_scalar_deriv.get(i) /
                    pow<5>(get(1 + conformal_factor_minus_one));
    result.get(i) += 2 * square(1 + get(conformal_factor_minus_one)) *
                     get(lapse_times_conformal_factor_correction) *
                     shift_rolloff.get(i) *
                     get(dot_product(shift_rolloff, scalar_deriv)) /
                     pow<3>(1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) -= 2 * (1 + get(conformal_factor_minus_one)) *
                     get(conformal_factor_correction) * shift_rolloff.get(i) *
                     get(dot_product(shift_rolloff, scalar_deriv)) /
                     square(1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) -= square(1 + get(conformal_factor_minus_one)) *
                     get(lapse_times_conformal_factor_correction) *
                     shift_excess_correction.get(i) *
                     get(dot_product(shift_rolloff, scalar_deriv)) /
                     square(1 + get(lapse_times_conformal_factor_minus_one));
    result.get(i) -= square(1 + get(conformal_factor_minus_one)) *
                     get(lapse_times_conformal_factor_correction) *
                     shift_rolloff.get(i) *
                     get(dot_product(shift_excess_correction, scalar_deriv)) /
                     square(1 + get(lapse_times_conformal_factor_minus_one));
  }
  return result;
}

void add_GB_terms(gsl::not_null<Scalar<DataVector>*> scalar_tensor_equation,
                  const double eps2, const double eps4,
                  const Scalar<DataVector>& weyl_electric,
                  const Scalar<DataVector>& weyl_magnetic,
                  const Scalar<DataVector>& field) {
  get(*scalar_tensor_equation) -=
      2. * (weyl_electric.get() - weyl_magnetic.get()) *
      (eps2 * field.get() + eps4 * cube(get(field)));
}

void add_linearized_GB_terms(
    gsl::not_null<Scalar<DataVector>*> linearized_scalar_tensor_equation,
    const double eps2, const double eps4,
    const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic, const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction) {
  get(*linearized_scalar_tensor_equation) -=
      2. * (weyl_electric.get() - weyl_magnetic.get()) *
      (eps2 * field_correction.get() +
       3. * eps4 * square(get(field)) * field_correction.get());
}

void Fluxes::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor,
    gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
    const double& rolloff_location, const double& rolloff_rate,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const Scalar<DataVector>& scalar,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
    const tnsr::i<DataVector, 3>& scalar_gradient) {
  ::Xcts::Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
      flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
      longitudinal_shift_excess, conformal_metric, inv_conformal_metric,
      christoffel_second_kind, conformal_factor_minus_one,
      lapse_times_conformal_factor_minus_one, shift_excess,
      conformal_factor_gradient, lapse_times_conformal_factor_gradient,
      deriv_shift_excess);
  tnsr::I<DataVector, 3> shift_rolloff;
  for (size_t i = 0; i < 3; i++) {
    shift_rolloff.get(i) = shift_excess.get(i) +
                           (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                               shift_background / 2;
  }
  curved_fluxes(flux_for_scalar, inv_conformal_metric, shift_rolloff,
                lapse_times_conformal_factor_minus_one,
                conformal_factor_minus_one, scalar_gradient, rolloff_rate,
                rolloff_location);
}

void Fluxes::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor,
    gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
    const double& rolloff_location, const double& rolloff_rate,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::I<DataVector, 3>& coordinates,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const Scalar<DataVector>& scalar) {
  ::Xcts::Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
      flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
      longitudinal_shift_excess, conformal_metric, inv_conformal_metric,
      christoffel_second_kind, face_normal, face_normal_vector,
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess);
  tnsr::I<DataVector, 3> shift_rolloff;
  for (size_t i = 0; i < 3; i++) {
    shift_rolloff.get(i) = shift_excess.get(i) +
                           (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                               shift_background / 2;
  }
  face_fluxes(flux_for_scalar, inv_conformal_metric, shift_rolloff,
              lapse_times_conformal_factor_minus_one,
              conformal_factor_minus_one, face_normal, scalar);
}

void LinearizedFluxes::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor_correction,
    gsl::not_null<tnsr::II<DataVector, 3>*>
        longitudinal_shift_excess_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar_correction,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
    const double& rolloff_location, const double& rolloff_rate,
    const tnsr::I<DataVector, 3>& shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_excess_correction,
    const Scalar<DataVector>& scalar_correction,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient_correction,
    const tnsr::i<DataVector, 3>&
        lapse_times_conformal_factor_gradient_correction,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
    const tnsr::i<DataVector, 3>& scalar_gradient_correction) {
  ::Xcts::Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
      flux_for_conformal_factor_correction,
      flux_for_lapse_times_conformal_factor_correction,
      longitudinal_shift_excess_correction, conformal_metric,
      inv_conformal_metric, christoffel_second_kind, conformal_factor_minus_one,
      shift_excess, conformal_factor_gradient,
      lapse_times_conformal_factor_gradient, deriv_shift_excess);
  tnsr::I<DataVector, 3> shift_rolloff;
  for (size_t i = 0; i < 3; i++) {
    shift_rolloff.get(i) = shift_excess.get(i) +
                           (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                               shift_background / 2;
  }
  curved_fluxes(flux_for_scalar_correction, inv_conformal_metric, shift_rolloff,
                lapse_times_conformal_factor_minus_one,
                conformal_factor_minus_one, scalar_gradient_correction,
                rolloff_rate, rolloff_location);
}

void LinearizedFluxes::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor_correction,
    gsl::not_null<tnsr::II<DataVector, 3>*>
        longitudinal_shift_excess_correction,
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_scalar_correction,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& christoffel_second_kind,
    const double& rolloff_location, const double& rolloff_rate,
    const tnsr::I<DataVector, 3>& shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const tnsr::I<DataVector, 3>& coordinates,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_excess_correction,
    const Scalar<DataVector>& scalar_correction) {
  ::Xcts::Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
      flux_for_conformal_factor_correction,
      flux_for_lapse_times_conformal_factor_correction,
      longitudinal_shift_excess_correction, conformal_metric,
      inv_conformal_metric, christoffel_second_kind, face_normal,
      face_normal_vector, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_excess_correction);
  tnsr::I<DataVector, 3> shift_rolloff;
  for (size_t i = 0; i < 3; i++) {
    shift_rolloff.get(i) = shift_excess.get(i) +
                           (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                               shift_background / 2;
  }
  face_fluxes(flux_for_scalar_correction, inv_conformal_metric, shift_rolloff,
              lapse_times_conformal_factor_minus_one,
              conformal_factor_minus_one, face_normal, scalar_correction);
}

void Sources::apply(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    gsl::not_null<Scalar<DataVector>*> scalar_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_background_minus_dt_conformal_metric,
    const tnsr::I<DataVector, 3>&
        div_longitudinal_shift_background_minus_dt_conformal_metric,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const Scalar<DataVector>& conformal_ricci_scalar, const double& eps2,
    const double& eps4, const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const Scalar<DataVector>& scalar,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
    const tnsr::I<DataVector, 3>& scalar_flux) {
  add_curved_sources(scalar_equation, conformal_metric,
                     conformal_christoffel_contracted, scalar_flux,
                     lapse_times_conformal_factor_flux,
                     lapse_times_conformal_factor_minus_one,
                     conformal_factor_minus_one, conformal_factor_flux);
  add_GB_terms(scalar_equation, eps2, eps4, weyl_electric, weyl_magnetic,
               scalar);
  ::Xcts::Sources<Equations::HamiltonianLapseAndShift, Geometry::Curved, 0>::
      apply(hamiltonian_constraint, lapse_equation, momentum_constraint,
            conformal_energy_density, conformal_stress_trace,
            conformal_momentum_density, extrinsic_curvature_trace,
            dt_extrinsic_curvature_trace, extrinsic_curvature_trace_gradient,
            shift_background,
            longitudinal_shift_background_minus_dt_conformal_metric,
            div_longitudinal_shift_background_minus_dt_conformal_metric,
            conformal_metric, inv_conformal_metric,
            conformal_christoffel_first_kind, conformal_christoffel_second_kind,
            conformal_christoffel_contracted, conformal_ricci_scalar,
            conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
            shift_excess, conformal_factor_flux,
            lapse_times_conformal_factor_flux, longitudinal_shift_excess)
}

void LinearizedSources::apply(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_scalar_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_background_minus_dt_conformal_metric,
    const tnsr::I<DataVector, 3>&
        div_longitudinal_shift_background_minus_dt_conformal_metric,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const Scalar<DataVector>& conformal_ricci_scalar, const double& eps2,
    const double& eps4, const Scalar<DataVector>& weyl_electric,
    const Scalar<DataVector>& weyl_magnetic,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& shift_excess,
    const Scalar<DataVector>& scalar,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
    const tnsr::I<DataVector, 3>& scalar_flux, const double& rolloff_location,
    const double& rolloff_rate, const tnsr::I<DataVector, 3>& coordinates,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_excess_correction,
    const Scalar<DataVector>& scalar_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_excess_correction,
    const tnsr::I<DataVector, 3>& scalar_flux_correction) {
  ::Xcts::LinearizedSources<
      Equations::HamiltonianLapseAndShift, Geometry::Curved,
      0>::apply(linearized_hamiltonian_constraint, linearized_lapse_equation,
                linearized_momentum_constraint, conformal_energy_density,
                conformal_stress_trace, conformal_momentum_density,
                extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
                extrinsic_curvature_trace_gradient, shift_background,
                longitudinal_shift_background_minus_dt_conformal_metric,
                div_longitudinal_shift_background_minus_dt_conformal_metric,
                conformal_metric, inv_conformal_metric,
                conformal_christoffel_first_kind,
                conformal_christoffel_second_kind,
                conformal_christoffel_contracted, conformal_ricci_scalar,
                conformal_factor_minus_one,
                lapse_times_conformal_factor_minus_one, shift_excess,
                conformal_factor_flux, lapse_times_conformal_factor_flux,
                longitudinal_shift_excess, conformal_factor_correction,
                lapse_times_conformal_factor_correction,
                shift_excess_correction, conformal_factor_flux_correction,
                lapse_times_conformal_factor_flux_correction,
                longitudinal_shift_excess_correction);
  tnsr::I<DataVector, 3> shift_rolloff;
  for (size_t i = 0; i < 3; i++) {
    shift_rolloff.get(i) = shift_excess.get(i) +
                           (1 - tanh((r - rolloff_location) * rolloff_rate)) *
                               shift_background / 2;
  }
  const tnsr::I<DataVector, 3> linearized_fluxes = flux_part_linearization(
      mesh, inv_jacobian, inv_conformal_metric, scalar,
      lapse_times_conformal_factor_minus_one, conformal_factor_minus_one,
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_rolloff, shift_excess_correction);
  const tnsr::I<DataVector, 3> linearized_sources = source_part_linearization(
      lapse_times_conformal_factor_flux, lapse_times_conformal_factor_minus_one,
      conformal_factor_minus_one, conformal_factor_flux,
      conformal_factor_correction, lapse_times_conformal_factor_correction);
  const tnsr::I<DataVector, 3> sources = curved_sources(
      conformal_metric, conformal_christoffel_contracted, scalar_flux,
      lapse_times_conformal_factor_flux, lapse_times_conformal_factor_minus_one,
      conformal_factor_minus_one, conformal_factor_flux);
  const auto derivative_flux_terms =
      partial_derivative(linearized_fluxes, mesh, inv_jacobian);
  for (size_t i = 0; i < 3; i++) {
    get(*linearized_scalar_equation) += derivative_flux_terms.get(i, i);
  }
  const auto lowered_source = raise_or_lower_index(sources, conformal_metric);
  add_curved_sources(linearized_scalar_equation, conformal_metric,
                     conformal_christoffel_contracted, scalar_flux_correction,
                     lapse_times_conformal_factor_flux,
                     lapse_times_conformal_factor_minus_one,
                     conformal_factor_minus_one, conformal_factor_flux);
  get(*linearized_scalar_equation) -=
      get(dot_product(sources, linearized_fluxes));
  get(*linearized_scalar_equation) -= get(dot_product(sources, scalar_flux));
}

}  // namespace sgb
