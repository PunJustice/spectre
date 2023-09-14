// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Cowling/KerrSchildCowling.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Cowling::Solutions {
KerrSchild::KerrSchild(CkMigrateMessage* /*msg*/) {}

KerrSchild::KerrSchild(const double mass,
                       const std::array<double, 3>& dimensionless_spin,
                       const std::array<double, 3>& center)
    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      dimensionless_spin_(dimensionless_spin),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(center),
      zero_spin_(dimensionless_spin_ == std::array<double, 3>{{0., 0., 0.}}) {}

void KerrSchild::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
  p | zero_spin_;
}

PUP::able::PUP_ID KerrSchild::my_PUP_ID = 0;  // NOLINT

std::unique_ptr<elliptic::analytic_data::AnalyticSolution>
KerrSchild::get_clone() const {
  return std::make_unique<KerrSchild>(mass_, dimensionless_spin_, center_);
}

template <typename DataType, typename Frame>
KerrSchild::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const KerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x,
    const std::optional<std::reference_wrapper<const Mesh<3>>>& mesh,
    const std::optional<std::reference_wrapper<const InverseJacobian<
        DataType, 3, ::Frame::ElementLogical, ::Frame::Inertial>>>&
        inv_jacobian)
    : solution_(solution), x_(x), mesh_(mesh), inv_jacobian_(inv_jacobian) {}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_minus_center<DataType, Frame> /*meta*/) const {
  *x_minus_center = x_;
  for (size_t d = 0; d < 3; ++d) {
    x_minus_center->get(d) -= gsl::at(solution_.center(), d);
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  get(*a_dot_x) = spin_a[0] * get<0>(x_minus_center) +
                  spin_a[1] * get<1>(x_minus_center) +
                  spin_a[2] * get<2>(x_minus_center);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x_squared<DataType> /*meta*/) const {
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*a_dot_x_squared) = square(a_dot_x);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> half_xsq_minus_asq,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::half_xsq_minus_asq<DataType> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  get(*half_xsq_minus_asq) =
      0.5 * (square(get<0>(x_minus_center)) + square(get<1>(x_minus_center)) +
             square(get<2>(x_minus_center)) - a_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const {
  if (solution_.zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    dot_product(r_squared, x_minus_center, x_minus_center);
    return;
  }
  const auto& half_xsq_minus_asq =
      get(cache->get_var(*this, internal_tags::half_xsq_minus_asq<DataType>{}));
  const auto& a_dot_x_squared =
      get(cache->get_var(*this, internal_tags::a_dot_x_squared<DataType>{}));

  get(*r_squared) =
      half_xsq_minus_asq + sqrt(square(half_xsq_minus_asq) + a_dot_x_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  get(*r) = sqrt(r_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x_over_rsquared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x_over_rsquared<DataType> /*meta*/) const {
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  get(*a_dot_x_over_rsquared) = a_dot_x / r_squared;
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_log_r_denom,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_log_r_denom<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& half_xsq_minus_asq =
      get(cache->get_var(*this, internal_tags::half_xsq_minus_asq<DataType>{}));

  get(*deriv_log_r_denom) = 0.5 / (r_squared - half_xsq_minus_asq);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_log_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_log_r<DataType, Frame> /*meta*/) const {
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& a_dot_x_over_rsquared = get(
      cache->get_var(*this, internal_tags::a_dot_x_over_rsquared<DataType>{}));
  const auto& deriv_log_r_denom =
      get(cache->get_var(*this, internal_tags::deriv_log_r_denom<DataType>{}));

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  for (size_t i = 0; i < 3; ++i) {
    deriv_log_r->get(i) =
        deriv_log_r_denom *
        (x_minus_center.get(i) + gsl::at(spin_a, i) * a_dot_x_over_rsquared);
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> H_denom,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::H_denom<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& a_dot_x_squared =
      get(cache->get_var(*this, internal_tags::a_dot_x_squared<DataType>{}));

  get(*H_denom) = 1.0 / (square(r_squared) + a_dot_x_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::H<DataType> /*meta*/) const {
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& H_denom =
      get(cache->get_var(*this, internal_tags::H_denom<DataType>{}));

  get(*H) = solution_.mass() * r * r_squared * H_denom;
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_H_temp1,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H_temp1<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto& H_denom =
      get(cache->get_var(*this, internal_tags::H_denom<DataType>{}));

  get(*deriv_H_temp1) = H * (3.0 - 4.0 * square(r_squared) * H_denom);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_H_temp2,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H_temp2<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& H_denom =
      get(cache->get_var(*this, internal_tags::H_denom<DataType>{}));
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

  get(*deriv_H_temp2) = H * (2.0 * H_denom * a_dot_x);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_H,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_H<DataType, Frame> /*meta*/) const {
  const auto& deriv_log_r =
      cache->get_var(*this, internal_tags::deriv_log_r<DataType, Frame>{});
  const auto& deriv_H_temp1 =
      get(cache->get_var(*this, internal_tags::deriv_H_temp1<DataType>{}));
  const auto& deriv_H_temp2 =
      get(cache->get_var(*this, internal_tags::deriv_H_temp2<DataType>{}));

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  for (size_t i = 0; i < 3; ++i) {
    deriv_H->get(i) =
        deriv_H_temp1 * deriv_log_r.get(i) - deriv_H_temp2 * gsl::at(spin_a, i);
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> denom,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::denom<DataType> /*meta*/) const {
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  get(*denom) = 1.0 / (r_squared + a_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> a_dot_x_over_r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::a_dot_x_over_r<DataType> /*meta*/) const {
  const auto& a_dot_x =
      get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  get(*a_dot_x_over_r) = a_dot_x / r;
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> null_form,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::null_form<DataType, Frame> /*meta*/) const {
  const auto& a_dot_x_over_r =
      get(cache->get_var(*this, internal_tags::a_dot_x_over_r<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& denom =
      get(cache->get_var(*this, internal_tags::denom<DataType>{}));

  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  for (size_t i = 0; i < 3; ++i) {
    const size_t cross_product_index_1 = (i + 1) % 3;
    const size_t cross_product_index_2 = (i + 2) % 3;
    null_form->get(i) = denom * (r * x_minus_center.get(i) -
                                 gsl::at(spin_a, cross_product_index_1) *
                                     x_minus_center.get(cross_product_index_2) +
                                 gsl::at(spin_a, cross_product_index_2) *
                                     x_minus_center.get(cross_product_index_1) +
                                 a_dot_x_over_r * gsl::at(spin_a, i));
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ij<DataType, 3, Frame>*> deriv_null_form,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_null_form<DataType, Frame> /*meta*/) const {
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& denom =
      get(cache->get_var(*this, internal_tags::denom<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
  const auto& x_minus_center =
      cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});
  const auto& a_dot_x_over_rsquared = get(
      cache->get_var(*this, internal_tags::a_dot_x_over_rsquared<DataType>{}));
  const auto& deriv_log_r =
      cache->get_var(*this, internal_tags::deriv_log_r<DataType, Frame>{});

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      deriv_null_form->get(j, i) =
          denom * (gsl::at(spin_a, i) * gsl::at(spin_a, j) / r +
                   (x_minus_center.get(i) - 2.0 * r * null_form.get(i) -
                    a_dot_x_over_rsquared * gsl::at(spin_a, i)) *
                       deriv_log_r.get(j) * r);
      if (i == j) {
        deriv_null_form->get(j, i) += denom * r;
      } else {         //  add denom*epsilon^ijk a_k
        size_t k = (j + 1) % 3;
        if (k == i) {  // j+1 = i (cyclic), so choose minus sign
          k++;
          k = k % 3;   // and set k to be neither i nor j
          deriv_null_form->get(j, i) -= denom * gsl::at(spin_a, k);
        } else {       // i+1 = j (cyclic), so choose plus sign
          deriv_null_form->get(j, i) += denom * gsl::at(spin_a, k);
        }
      }
    }
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::lapse_squared<DataType> /*meta*/) const {
  if (solution_.zero_spin()) {
    const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
    get(*lapse_squared) = 1.0 / (1.0 + 2.0 * solution_.mass() / r);
    return;
  }
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  get(*lapse_squared) = 1.0 / (1.0 + 2.0 * square(null_vector_0_) * H);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  get(*lapse) = sqrt(lapse_squared);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const {
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  get(*deriv_lapse_multiplier) =
      -square(null_vector_0_) * lapse * lapse_squared;
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> shift_multiplier,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::shift_multiplier<DataType> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));

  get(*shift_multiplier) = -2.0 * null_vector_0_ * H * lapse_squared;
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const {
  if (solution_.zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    const auto& r_squared =
        get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
    const auto& lapse_squared =
        get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
    for (size_t i = 0; i < 3; ++i) {
      shift->get(i) = 2.0 * solution_.mass() * x_minus_center.get(i) *
                      lapse_squared / r_squared;
    }
    return;
  }
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});
  const auto& shift_multiplier =
      get(cache->get_var(*this, internal_tags::shift_multiplier<DataType>{}));

  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_multiplier * null_form.get(i);
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    const gsl::not_null<CachedBuffer*> cache,
    DerivShift<DataType, Frame> /*meta*/) const {
  if (solution_.zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
    const auto& r_squared =
        get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
    const auto& lapse_squared =
        get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        deriv_shift->get(i, j) =
            (-4.0 * solution_.mass() * x_minus_center.get(i) *
             x_minus_center.get(j) * lapse_squared *
             (1 - solution_.mass() * lapse_squared / r)) /
            square(r_squared);
      }
      deriv_shift->get(i, i) +=
          2.0 * solution_.mass() * lapse_squared / r_squared;
    }
    return;
  }
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& deriv_null_form =
      cache->get_var(*this, internal_tags::deriv_null_form<DataType, Frame>{});

  for (size_t m = 0; m < 3; ++m) {
    for (size_t i = 0; i < 3; ++i) {
      deriv_shift->get(m, i) = 4.0 * cube(null_vector_0_) * H *
                                   null_form.get(i) * square(lapse_squared) *
                                   deriv_H.get(m) -
                               2.0 * null_vector_0_ * lapse_squared *
                                   (null_form.get(i) * deriv_H.get(m) +
                                    H * deriv_null_form.get(m, i));
    }
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const {
  if (solution_.zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
    const auto& r_squared =
        get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        spatial_metric->get(i, j) = 2.0 * solution_.mass() *
                                    x_minus_center.get(i) *
                                    x_minus_center.get(j) / r_squared / r;
      }
      spatial_metric->get(i, i) += 1.0;
    }
    return;
  }
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});

  std::fill(spatial_metric->begin(), spatial_metric->end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric->get(i, i) = 1.;
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      spatial_metric->get(i, j) +=
          2.0 * H * null_form.get(i) * null_form.get(j);
    }
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame>*> inverse_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) const {
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& lapse_squared =
      get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});
  if (solution_.zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
    const auto& r_squared =
        get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        inverse_spatial_metric->get(i, j) =
            -2.0 * solution_.mass() * x_minus_center.get(i) *
            x_minus_center.get(j) /
            (2 * solution_.mass() * r_squared + r_squared * r);
      }
      inverse_spatial_metric->get(i, i) += 1.0;
    }
    return;
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      inverse_spatial_metric->get(i, j) =
          -2.0 * H * lapse_squared * null_form.get(i) * null_form.get(j);
    }
    inverse_spatial_metric->get(i, i) += 1.;
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    DerivSpatialMetric<DataType, Frame> /*meta*/) const {
  const auto& null_form =
      cache->get_var(*this, internal_tags::null_form<DataType, Frame>{});
  const auto& deriv_H =
      cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
  const auto& H = get(cache->get_var(*this, internal_tags::H<DataType>{}));
  const auto& deriv_null_form =
      cache->get_var(*this, internal_tags::deriv_null_form<DataType, Frame>{});

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      for (size_t m = 0; m < 3; ++m) {
        deriv_spatial_metric->get(m, i, j) =
            2.0 * null_form.get(i) * null_form.get(j) * deriv_H.get(m) +
            2.0 * H *
                (null_form.get(i) * deriv_null_form.get(m, j) +
                 null_form.get(j) * deriv_null_form.get(m, i));
      }
    }
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const {
  std::fill(dt_spatial_metric->begin(), dt_spatial_metric->end(), 0.);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> extrinsic_curvature,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) const {
  gr::extrinsic_curvature(
      extrinsic_curvature, cache->get_var(*this, gr::Tags::Lapse<DataType>{}),
      cache->get_var(*this, gr::Tags::Shift<DataType, 3, Frame>{}),
      cache->get_var(*this, DerivShift<DataType, Frame>{}),
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3, Frame>{}),
      cache->get_var(*this,
                     ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>{}),
      cache->get_var(*this, DerivSpatialMetric<DataType, Frame>{}));
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
        spatial_christoffel_first_kind,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame> /*meta*/) const {
  const auto& d_spatial_metric =
      cache->get_var(*this, DerivSpatialMetric<DataType, Frame>{});
  gr::christoffel_first_kind<3, Frame, IndexType::Spatial, DataType>(
      spatial_christoffel_first_kind, d_spatial_metric);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ijj<DataType, 3, Frame>*>
        spatial_christoffel_second_kind,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/) const {
  const auto& spatial_christoffel_first_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>{});
  const auto& inverse_spatial_metric = cache->get_var(
      *this, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});
  raise_or_lower_first_index<DataType, SpatialIndex<3, UpLo::Lo, Frame>,
                             SpatialIndex<3, UpLo::Lo, Frame>>(
      spatial_christoffel_second_kind, spatial_christoffel_first_kind,
      inverse_spatial_metric);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::iJkk<DataType, 3, Frame>*>
        deriv_spatial_christoffel_second_kind,
    const gsl::not_null<CachedBuffer*> cache,
    ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
                  tmpl::size_t<3>, ::Frame::Inertial> /*meta*/) const {
  const auto& spatial_christoffel_second_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  partial_derivative(deriv_spatial_christoffel_second_kind,
                     spatial_christoffel_second_kind, mesh_->get(),
                     inv_jacobian_->get());
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*>
        spatial_christoffel_second_kind_contracted,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialChristoffelSecondKindContracted<DataType, 3,
                                                     Frame> /*meta*/) const {
  const auto& spatial_christoffel_second_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  for (size_t i = 0; i < 3; ++i) {
    spatial_christoffel_second_kind_contracted->get(i) = 0. * get<0>(x_);
    for (size_t j = 0; j < 3; ++j) {
      spatial_christoffel_second_kind_contracted->get(i) +=
          spatial_christoffel_second_kind.get(j, j, i);
    }
  }
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_ricci,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SpatialRicci<DataType, 3, Frame> /*meta*/) const {
  const auto& spatial_christoffel_second_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  const auto& deriv_spatial_christoffel_second_kind = cache->get_var(
      *this,
      ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
                    tmpl::size_t<3>, ::Frame::Inertial>{});
  gr::ricci_tensor(spatial_ricci, spatial_christoffel_second_kind,
                   deriv_spatial_christoffel_second_kind);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::WeylElectricScalar<DataType> /*meta*/) const {
  const auto& spatial_ricci =
      cache->get_var(*this, gr::Tags::SpatialRicci<DataType, 3, Frame>{});
  const auto& extrinsic_curvature =
      cache->get_var(*this, gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>{});
  const auto& inverse_spatial_metric = cache->get_var(
      *this, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});

  const auto weyl_electric = gr::weyl_electric(
      spatial_ricci, extrinsic_curvature, inverse_spatial_metric);
  gr::weyl_electric_scalar(weyl_electric_scalar, weyl_electric,
                           inverse_spatial_metric);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::WeylMagneticScalar<DataType> /*meta*/) const {
  const auto& extrinsic_curvature =
      cache->get_var(*this, gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3, Frame>{});
  const auto& sqrt_det_spatial_metric =
      cache->get_var(*this, gr::Tags::SqrtDetSpatialMetric<DataType>{});
  const auto& spatial_christoffel_second_kind = cache->get_var(
      *this, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  const auto& inverse_spatial_metric = cache->get_var(
      *this, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});

  const auto deriv_extrinsic_curvature = partial_derivative(
      extrinsic_curvature, mesh_->get(), inv_jacobian_->get());
  tnsr::ijj<DataType, 3, Frame> cov_deriv_extrinsic_curvature{};
  tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&cov_deriv_extrinsic_curvature),
      deriv_extrinsic_curvature(ti::i, ti::j, ti::k) -
          spatial_christoffel_second_kind(ti::L, ti::i, ti::j) *
              extrinsic_curvature(ti::l, ti::k) -
          spatial_christoffel_second_kind(ti::L, ti::i, ti::k) *
              extrinsic_curvature(ti::j, ti::l));
  const auto weyl_magnetic = gr::weyl_magnetic(
      cov_deriv_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric);
  gr::weyl_magnetic_scalar(weyl_magnetic_scalar, weyl_magnetic,
                           inverse_spatial_metric);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> sqrt_det_spatial_metric,
    const gsl::not_null<CachedBuffer*> cache,
    gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) const {
  const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
  sqrt_det_spatial_metric->get() = (1.0 / lapse);
}

template <typename DataType, typename Frame>
void KerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_lapse,
    const gsl::not_null<CachedBuffer*> cache,
    DerivLapse<DataType, Frame> /*meta*/) const {
  if (solution().zero_spin()) {
    const auto& x_minus_center =
        cache->get_var(*this, internal_tags::x_minus_center<DataType, Frame>{});
    const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
    const auto& r_squared =
        get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

    const auto& lapse = get(cache->get_var(*this, gr::Tags::Lapse<DataType>{}));
    const auto& lapse_squared =
        get(cache->get_var(*this, internal_tags::lapse_squared<DataType>{}));
    for (size_t i = 0; i < 3; ++i) {
      deriv_lapse->get(i) = solution().mass() * x_minus_center.get(i) * lapse *
                            lapse_squared / r_squared / r;
    }
  } else {
    const auto& deriv_H =
        cache->get_var(*this, internal_tags::deriv_H<DataType, Frame>{});
    const auto& deriv_lapse_multiplier = get(cache->get_var(
        *this, internal_tags::deriv_lapse_multiplier<DataType>{}));

    for (size_t i = 0; i < 3; ++i) {
      deriv_lapse->get(i) = deriv_lapse_multiplier * deriv_H.get(i);
    }
  }
}

template <typename DataType, typename Frame>
Scalar<DataType> KerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>{}));
  return make_with_value<Scalar<DataType>>(H, 0.);
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
KerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/) {
  const auto& H = get(get_var(computer, internal_tags::H<DataType>()));
  return make_with_value<tnsr::I<DataType, 3, Frame>>(H, 0.);
}

template <typename DataType, typename Frame>
tnsr::i<DataType, 3, Frame>
KerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame> /*meta*/) {
  const auto& deriv_H =
      get_var(computer, internal_tags::deriv_H<DataType, Frame>{});

  auto result =
      make_with_value<tnsr::i<DataType, 3, Frame>>(get<0>(deriv_H), 0.);
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) = 2.0 * square(null_vector_0_) * deriv_H.get(i);
  }

  return result;
}

template <typename DataType, typename Frame>
Scalar<DataType> KerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) {
  if (computer.solution().zero_spin()) {
    const double m = computer.solution().mass();
    const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
    Scalar<DataType> result(get_size(r));
    get(result) = 1. / ((2. * m + r) * r);
    get(result) = sqrt(cube(result.get()));
    get(result) *= 2. * m * (3. * m + r);
    return result;
  }
  return trace(
      get_var(computer, gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>{}),
      get_var(computer, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{}));
}

template <typename DataType, typename Frame>
tnsr::I<DataType, 3, Frame>
KerrSchild::IntermediateVars<DataType, Frame>::get_var(
    const IntermediateComputer<DataType, Frame>& computer,
    gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/) {
  if (computer.solution().zero_spin()) {
    const double m = computer.solution().mass();
    const auto& r = get(get_var(computer, internal_tags::r<DataType>{}));
    const auto& r_squared =
        get(get_var(computer, internal_tags::r_squared<DataType>{}));
    const auto& x_minus_center =
        get_var(computer, internal_tags::x_minus_center<DataType, Frame>{});
    DataType factor = m * (8. * m + 3. * r) / square(2. * m + r) / r_squared;
    tnsr::I<DataType, 3, Frame> result(get_size(r));
    for (size_t i = 0; i < 3; ++i) {
      result.get(i) = factor * x_minus_center.get(i);
    }
    return result;
  }
  const auto& inverse_spatial_metric =
      get_var(computer, gr::Tags::InverseSpatialMetric<DataType, 3, Frame>{});
  const auto& spatial_christoffel_second_kind = get_var(
      computer, gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>{});
  return trace_last_indices<DataType, SpatialIndex<3, UpLo::Up, Frame>,
                            SpatialIndex<3, UpLo::Lo, Frame>>(
      spatial_christoffel_second_kind, inverse_spatial_metric);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                             \
  template class KerrSchild::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class KerrSchild::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector), (::Frame::Inertial))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace Cowling::Solutions