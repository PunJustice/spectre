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

template <size_t Dim>
void flat_cartesian_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) {
  for (size_t d = 0; d < Dim; d++) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <size_t Dim>
void curved_fluxes(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) {
  raise_or_lower_index(flux_for_field, field_gradient, inv_spatial_metric);
}

template <size_t Dim>
void add_curved_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::I<DataVector, Dim>& flux_for_field,
    const tnsr::i<DataVector, Dim>& deriv_lapse,
    const Scalar<DataVector>& lapse) {
  get(*source_for_field) = get(dot_product(deriv_lapse, flux_for_field));
  get(*source_for_field) /= get(lapse);
  get(*source_for_field) -=
      get(dot_product(christoffel_contracted, flux_for_field));
}

template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field) {
  std::fill(flux_for_gradient->begin(), flux_for_gradient->end(), 0.);
  for (size_t d = 0; d < Dim; d++) {
    flux_for_gradient->get(d, d) = get(field);
  }
}

template <size_t Dim>
void Fluxes<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) {
  flat_cartesian_fluxes(flux_for_field, field_gradient);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) {
  curved_fluxes(flux_for_field, inv_spatial_metric, field_gradient);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field) {
  auxiliary_fluxes(flux_for_gradient, field);
}

template <size_t Dim>
void Fluxes<Dim, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
    const Scalar<DataVector>& field) {
  auxiliary_fluxes(flux_for_gradient, field);
}

template <size_t Dim>
void Sources<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<Scalar<DataVector>*> /*equation_for_field*/,
    const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, Dim>& /*field_flux*/) {}

template <size_t Dim>
void Sources<Dim, Geometry::Curved>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::i<DataVector, Dim>& deriv_lapse,
    const Scalar<DataVector>& lapse, const Scalar<DataVector>& /*field*/,
    const tnsr::I<DataVector, Dim>& field_flux) {
  add_curved_sources(equation_for_field, christoffel_contracted, field_flux,
                     deriv_lapse, lapse);
}

template <size_t Dim>
void Sources<Dim, Geometry::FlatCartesian>::apply(
    const gsl::not_null<
        tnsr::i<DataVector, Dim>*> /*equation_for_field_gradient*/,
    const Scalar<DataVector>& /*field*/) {}

template <size_t Dim>
void Sources<Dim, Geometry::Curved>::apply(
    const gsl::not_null<
        tnsr::i<DataVector, Dim>*> /*equation_for_field_gradient*/,
    const tnsr::i<DataVector, Dim>& /*christoffel_contracted*/,
    const tnsr::i<DataVector, Dim>& /*deriv_lapse*/,
    const Scalar<DataVector>& /*lapse*/, const Scalar<DataVector>& /*field*/) {}

}  // namespace Cowling

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void Cowling::flat_cartesian_fluxes<DIM(data)>(                     \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::i<DataVector, DIM(data)>&);                                  \
  template void Cowling::curved_fluxes<DIM(data)>(                             \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,                    \
      const tnsr::II<DataVector, DIM(data)>&,                                  \
      const tnsr::i<DataVector, DIM(data)>&);                                  \
  template void Cowling::add_curved_sources<DIM(data)>(                        \
      const gsl::not_null<Scalar<DataVector>*>,                                \
      const tnsr::i<DataVector, DIM(data)>&,                                   \
      const tnsr::I<DataVector, DIM(data)>&,                                   \
      const tnsr::i<DataVector, DIM(data)>&, const Scalar<DataVector>&);       \
  template void Cowling::auxiliary_fluxes<DIM(data)>(                          \
      gsl::not_null<tnsr::Ij<DataVector, DIM(data)>*>,                         \
      const Scalar<DataVector>&);                                              \
  template class Cowling::Fluxes<DIM(data), Cowling::Geometry::FlatCartesian>; \
  template class Cowling::Fluxes<DIM(data), Cowling::Geometry::Curved>;        \
  template class Cowling::Sources<DIM(data),                                   \
                                  Cowling::Geometry::FlatCartesian>;           \
  template class Cowling::Sources<DIM(data), Cowling::Geometry::Curved>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
