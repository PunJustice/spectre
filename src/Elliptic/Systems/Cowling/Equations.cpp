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
                   const tnsr::II<DataVector, 3>& inv_spatial_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) -= get(dot_product(shift, field_gradient)) *
                              shift.get(i) / get(lapse) / get(lapse);
  }
}

void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::i<DataVector, 3>& deriv_lapse,
    const Scalar<DataVector>& lapse) {
  get(*source_for_field) -=
      get(dot_product(deriv_lapse, flux_for_field)) / get(lapse);
  get(*source_for_field) -=
      get(dot_product(christoffel_contracted, flux_for_field));
}

void face_fluxes(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                 const tnsr::II<DataVector, 3>& inv_spatial_metric,
                 const tnsr::I<DataVector, 3>& shift,
                 const Scalar<DataVector>& lapse,
                 const tnsr::i<DataVector, 3>& face_normal,
                 const Scalar<DataVector>& field) {
  raise_or_lower_index(flux_for_field, face_normal, inv_spatial_metric);
  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) -= get(dot_product(shift, face_normal)) *
                              shift.get(i) / get(lapse) / get(lapse);
    flux_for_field->get(i) *= get(field);
  }
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_spatial_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const Scalar<DataVector>& field,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  curved_fluxes(flux_for_field, inv_spatial_metric, shift, lapse,
                field_gradient);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> face_flux,
                   const tnsr::II<DataVector, 3>& inv_spatial_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const tnsr::i<DataVector, 3>& face_normal,
                   const tnsr::I<DataVector, 3>& face_normal_vector,
                   const Scalar<DataVector>& field) {
  face_fluxes(face_flux, inv_spatial_metric, shift, lapse, face_normal, field);
}

void Sources::apply(const gsl::not_null<Scalar<DataVector>*> equation_for_field,
                    const tnsr::i<DataVector, 3>& christoffel_contracted,
                    const tnsr::i<DataVector, 3>& deriv_lapse,
                    const Scalar<DataVector>& lapse,
                    const Scalar<DataVector>& /*field*/,
                    const tnsr::I<DataVector, 3>& field_flux) {
  add_curved_sources(equation_for_field, christoffel_contracted, field_flux,
                     deriv_lapse, lapse);
}

}  // namespace Cowling
