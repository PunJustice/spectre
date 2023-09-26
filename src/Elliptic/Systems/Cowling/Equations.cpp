// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Cowling/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Cowling {

void curved_fluxes(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_conformal_metric);
  for (size_t i = 0; i < 3; i++) {
    flux_for_field->get(i) -= get(dot_product(shift, field_gradient)) *
                              shift.get(i) / get(lapse) / get(lapse);
  }
}

void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_field,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv) {
  get(*source_for_field) -=
      get(dot_product(deriv_lapse, flux_for_field)) / get(lapse);
  get(*source_for_field) -=
      get(dot_product(conformal_christoffel_contracted, flux_for_field));
  get(*source_for_field) -=
      2 * get(dot_product(conformal_factor_deriv, flux_for_field));
}

void auxiliary_fluxes(gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_gradient,
                      const Scalar<DataVector>& field) {
  std::fill(flux_for_gradient->begin(), flux_for_gradient->end(), 0.);
  for (size_t d = 0; d < 3; d++) {
    flux_for_gradient->get(d, d) = get(field);
  }
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_field,
                   const tnsr::II<DataVector, 3>& inv_conformal_metric,
                   const tnsr::I<DataVector, 3>& shift,
                   const Scalar<DataVector>& lapse,
                   const tnsr::i<DataVector, 3>& field_gradient) {
  curved_fluxes(flux_for_field, inv_conformal_metric, shift, lapse,
                field_gradient);
}

void Fluxes::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_gradient,
    const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
    const tnsr::I<DataVector, 3>& /*shift*/,
    const Scalar<DataVector>& /*lapse*/, const Scalar<DataVector>& field) {
  auxiliary_fluxes(flux_for_gradient, field);
}

void Sources::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_field,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::i<DataVector, 3>& deriv_lapse, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, 3>& field_flux) {
  add_curved_sources(equation_for_field, conformal_christoffel_contracted,
                     field_flux, deriv_lapse, lapse, conformal_factor_deriv);
}

void Sources::apply(
    const gsl::not_null<
        tnsr::i<DataVector, 3>*> /*equation_for_field_gradient*/,
    const tnsr::i<DataVector, 3>& /*christoffel_contracted*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse*/,
    const Scalar<DataVector>& /*lapse*/,
    const tnsr::i<DataVector, 3>& /*conformal_factor_deriv*/,
    const Scalar<DataVector>& /*field*/) {}

}  // namespace Cowling
