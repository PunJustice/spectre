// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Cowling/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Cowling {

void curved_fluxes(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const double rolloff_location, const double rolloff_rate,
                   DataVector& r, const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_conformal_metric);

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= get(conformal_factor) / get(conformal_factor) /
                              get(conformal_factor) / get(conformal_factor);
    flux_for_field->get(i) -=
        (1 - tanh((r - rolloff_location) * rolloff_rate)) *
        get(dot_product(shift, field_gradient)) * shift.get(i) / get(lapse) /
        get(lapse) / 2;
  }
}

void face_fluxes(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_conformal_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse, const double rolloff_location,
                 const double rolloff_rate, DataVector& r,
                 const Scalar<DataVector>& conformal_factor,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field) {
  raise_or_lower_index(flux_for_field, face_normal, inv_conformal_metric);

  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) /= get(conformal_factor) / get(conformal_factor) /
                              get(conformal_factor) / get(conformal_factor);
    flux_for_field->get(i) -=
        (1 - tanh((r - rolloff_location) * rolloff_rate)) *
        get(dot_product(shift, face_normal)) * shift.get(i) / get(lapse) /
        get(lapse) / 2;
    flux_for_field->get(i) *= get(field);
  }
}

void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv) {
  get(*source_for_field) -=
      get(dot_product(deriv_lapse, flux_for_field)) / get(lapse);
  get(*source_for_field) -=
      get(dot_product(conformal_christoffel_contracted, flux_for_field));
  get(*source_for_field) -=
      6. * get(dot_product(conformal_factor_deriv, flux_for_field)) /
      get(conformal_factor);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const double rolloff_location, const double rolloff_rate,
                   const tnsr::I<DataVector, 3>& coords,
                   const Scalar<DataVector>& conformal_factor,
                   const Scalar<DataVector>& field,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  DataVector r = magnitude(coords).get();
  curved_fluxes(flux_for_field, inv_conformal_metric, shift, lapse,
                rolloff_location, rolloff_rate, r, conformal_factor,
                field_gradient);
}

void Fluxes::apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const double rolloff_location, const double rolloff_rate,
                   const tnsr::I<DataVector, 3>& coords,
                   const Scalar<DataVector>& conformal_factor,
                   const tnsr::i<DataVector, 3>& face_normal,
                   const tnsr::I<DataVector, 3>& face_normal_vector,
                   const Scalar<DataVector>& field) {
  DataVector r = magnitude(coords).get();
  face_fluxes(flux_for_field, inv_conformal_metric, shift, lapse,
              rolloff_location, rolloff_rate, r, conformal_factor, face_normal,
              field);
}

void Sources::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, 3>& field_flux) {
  add_curved_sources(equation_for_field, conformal_christoffel_contracted,
                     field_flux, deriv_lapse, lapse, conformal_factor,
                     conformal_factor_deriv);
}

}  // namespace Cowling
