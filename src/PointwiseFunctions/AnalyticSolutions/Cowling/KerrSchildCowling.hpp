// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cowling {
namespace Solutions {

/*!
 * \brief Kerr black hole in Kerr-Schild coordinates for Cowling solves.
 *
 * \details
 * Basically a copy of KerrSchild in gr::Solutions, but will be adapted to
 * interface properly with elliptic solver, and has a few more tags it can
 * compute.
 */
class KerrSchild : public ::gr::AnalyticSolution<3_st>,
                   public ::elliptic::analytic_data::Background {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() { return 0.; }
  };
  struct Spin {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] dimensionless spin of the black hole"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  struct Velocity {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] boost velocity of the black hole"};
  };
  using options = tmpl::list<Mass, Spin, Center, Velocity>;
  static constexpr Options::String help{
      "Black hole in Kerr-Schild coordinates"};

  KerrSchild(double mass, const std::array<double, 3>& dimensionless_spin,
             const std::array<double, 3>& center,
             const std::array<double, 3>& boost_velocity = {{0., 0., 0.}});

  /// \cond
  explicit KerrSchild(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KerrSchild);  // NOLINT
  /// \endcond

  template <typename DataType, typename Frame = Frame::Inertial>
  using tags = tmpl::flatten<tmpl::list<
      ::gr::AnalyticSolution<3_st>::tags<DataType, Frame>,
      gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame>,
      gr::Tags::TraceExtrinsicCurvature<DataType>,
      gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3, Frame>,
      gr::Tags::InverseSpatialMetric<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataType, 3, Frame>,
      gr::Tags::WeylElectricScalar<DataType>,
      gr::Tags::WeylMagneticScalar<DataType>,
      gr::Tags::SpatialMetric<DataType, 3, Frame>,
      gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>,
      gr::Tags::ShiftPlusVelocity<DataType, 3, Frame>,
      ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi>>>;

  KerrSchild() = default;
  KerrSchild(const KerrSchild& /*rhs*/) = default;
  KerrSchild& operator=(const KerrSchild& /*rhs*/) = default;
  KerrSchild(KerrSchild&& /*rhs*/) = default;
  KerrSchild& operator=(KerrSchild&& /*rhs*/) = default;
  ~KerrSchild() = default;
  std::unique_ptr<elliptic::analytic_data::Background> get_clone()
      const;

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, ::Frame::ElementLogical,
                            ::Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables_impl(
      const tnsr::I<DataType, volume_dim, Frame>& x,
      const std::optional<std::reference_wrapper<const Mesh<3>>>& mesh,
      const std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, ::Frame::ElementLogical, ::Frame::Inertial>>>&
          inv_jacobian,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x, mesh,
                                                   inv_jacobian);
    return {cache.get_var(computer, Tags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  double mass() const { return mass_; }
  const std::array<double, 3>& center() const { return center_; }
  const std::array<double, 3>& dimensionless_spin() const {
    return dimensionless_spin_;
  }
  const std::array<double, volume_dim>& boost_velocity() const {
    return boost_velocity_;
  }
  bool zero_spin() const { return zero_spin_; }
  bool zero_velocity() const { return zero_velocity_; }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center_unboosted = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_minus_center = ::Tags::TempI<1, 3, Frame, DataType>;
    template <typename DataType>
    using a_dot_x = ::Tags::TempScalar<2, DataType>;
    template <typename DataType>
    using a_dot_x_squared = ::Tags::TempScalar<3, DataType>;
    template <typename DataType>
    using half_xsq_minus_asq = ::Tags::TempScalar<4, DataType>;
    template <typename DataType>
    using r_squared = ::Tags::TempScalar<5, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<6, DataType>;
    template <typename DataType>
    using a_dot_x_over_rsquared = ::Tags::TempScalar<7, DataType>;
    template <typename DataType>
    using deriv_log_r_denom = ::Tags::TempScalar<8, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_log_r = ::Tags::Tempi<9, 3, Frame, DataType>;
    template <typename DataType>
    using H_denom = ::Tags::TempScalar<10, DataType>;
    template <typename DataType>
    using H = ::Tags::TempScalar<11, DataType>;
    template <typename DataType>
    using deriv_H_temp1 = ::Tags::TempScalar<12, DataType>;
    template <typename DataType>
    using deriv_H_temp2 = ::Tags::TempScalar<13, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H_unboosted = ::Tags::Tempa<14, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H = ::Tags::Tempa<15, 3, Frame, DataType>;
    template <typename DataType>
    using denom = ::Tags::TempScalar<16, DataType>;
    template <typename DataType>
    using a_dot_x_over_r = ::Tags::TempScalar<17, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using null_form_unboosted = ::Tags::Tempa<18, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using null_form = ::Tags::Tempa<19, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_null_form_unboosted = ::Tags::Tempab<20, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_null_form = ::Tags::Tempab<21, 3, Frame, DataType>;
    template <typename DataType>
    using null_form_dot_deriv_H = ::Tags::TempScalar<22, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using null_form_dot_deriv_null_form = ::Tags::Tempi<23, 3, Frame, DataType>;
    template <typename DataType>
    using lapse_squared = ::Tags::TempScalar<24, DataType>;
    template <typename DataType>
    using deriv_lapse_multiplier = ::Tags::TempScalar<25, DataType>;
    template <typename DataType>
    using shift_multiplier = ::Tags::TempScalar<26, DataType>;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      internal_tags::x_minus_center_unboosted<DataType, Frame>,
      internal_tags::x_minus_center<DataType, Frame>,
      internal_tags::a_dot_x<DataType>,
      internal_tags::a_dot_x_squared<DataType>,
      internal_tags::half_xsq_minus_asq<DataType>,
      internal_tags::r_squared<DataType>, internal_tags::r<DataType>,
      internal_tags::a_dot_x_over_rsquared<DataType>,
      internal_tags::deriv_log_r_denom<DataType>,
      internal_tags::deriv_log_r<DataType, Frame>,
      internal_tags::H_denom<DataType>, internal_tags::H<DataType>,
      internal_tags::deriv_H_temp1<DataType>,
      internal_tags::deriv_H_temp2<DataType>,
      internal_tags::deriv_H_unboosted<DataType, Frame>,
      internal_tags::deriv_H<DataType, Frame>, internal_tags::denom<DataType>,
      internal_tags::a_dot_x_over_r<DataType>,
      internal_tags::null_form_unboosted<DataType, Frame>,
      internal_tags::null_form<DataType, Frame>,
      internal_tags::deriv_null_form_unboosted<DataType, Frame>,
      internal_tags::deriv_null_form<DataType, Frame>,
      internal_tags::null_form_dot_deriv_H<DataType>,
      internal_tags::null_form_dot_deriv_null_form<DataType, Frame>,
      internal_tags::lapse_squared<DataType>, gr::Tags::Lapse<DataType>,
      internal_tags::deriv_lapse_multiplier<DataType>,
      internal_tags::shift_multiplier<DataType>,
      gr::Tags::Shift<DataType, 3, Frame>, DerivShift<DataType, Frame>,
      gr::Tags::ShiftPlusVelocity<DataType, 3, Frame>,
      gr::Tags::SpatialMetric<DataType, 3, Frame>,
      gr::Tags::InverseSpatialMetric<DataType, 3, Frame>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>,
      gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>,
      gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
      ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
                    tmpl::size_t<3>, ::Frame::Inertial>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataType, 3, Frame>,
      gr::Tags::SpatialRicci<DataType, 3, Frame>,
      gr::Tags::WeylElectricScalar<DataType>,
      gr::Tags::WeylMagneticScalar<DataType>,
      gr::Tags::SqrtDetSpatialMetric<DataType>, DerivLapse<DataType, Frame>>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = KerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(
        const KerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x,
        const std::optional<std::reference_wrapper<const Mesh<3>>>& mesh,
        const std::optional<std::reference_wrapper<const InverseJacobian<
            DataType, 3, ::Frame::ElementLogical, ::Frame::Inertial>>>&
            inv_jacobian);

    const KerrSchild& solution() const { return solution_; }

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center,
        const gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center_unboosted<DataType, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*>
            x_minus_center_boosted,
        const gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> half_xsq_minus_asq,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::half_xsq_minus_asq<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> a_dot_x_over_rsquared,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::a_dot_x_over_rsquared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_log_r_denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_log_r_denom<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_log_r,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_log_r<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> H_denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H_denom<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> H,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::H<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_H_temp1,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H_temp1<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> deriv_H_temp2,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::deriv_H_temp2<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::a<DataType, 3, Frame>*> deriv_H,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_H_unboosted<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::a<DataType, 3, Frame>*> deriv_H_boosted,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_H<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> denom,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::denom<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> a_dot_x_over_r,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x_over_r<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::a<DataType, 3, Frame>*> null_form,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::null_form_unboosted<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::a<DataType, 3, Frame>*> null_form_boosted,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::null_form<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ab<DataType, 3, Frame>*> deriv_null_form,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_null_form_unboosted<DataType, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::ab<DataType, 3, Frame>*>
            deriv_null_form_boosted,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_null_form<DataType, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> lapse_squared,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::lapse_squared<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> lapse,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Lapse<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_lapse_multiplier<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> shift_multiplier,
                    const gsl::not_null<CachedBuffer*> cache,
                    internal_tags::shift_multiplier<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::Shift<DataType, 3, Frame> /*meta*/) const;
    void operator()(
        const gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift_plus_velocity,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::ShiftPlusVelocity<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
        const gsl::not_null<CachedBuffer*> cache,
        DerivShift<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::II<DataType, 3, Frame>*> spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
                        deriv_spatial_metric,
                    const gsl::not_null<CachedBuffer*> cache,
                    DerivSpatialMetric<DataType, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> null_form_dot_deriv_H,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::null_form_dot_deriv_H<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::i<DataType, 3, Frame>*>
            null_form_dot_deriv_null_form,
        const gsl::not_null<CachedBuffer*> cache,
        internal_tags::null_form_dot_deriv_null_form<DataType, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> extrinsic_curvature,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*>
            spatial_christoffel_first_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::Ijj<DataType, 3, Frame>*>
            spatial_christoffel_second_kind,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame> /*meta*/)
        const;

    void operator()(
        const gsl::not_null<tnsr::iJkk<DataType, 3, Frame>*>
            deriv_spatial_christoffel_second_kind,
        const gsl::not_null<CachedBuffer*> cache,
        ::Tags::deriv<
            gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>,
            tmpl::size_t<3>, ::Frame::Inertial> /*meta*/) const;

    void operator()(const gsl::not_null<tnsr::i<DataType, 3, Frame>*>
                        spatial_christoffel_second_kind_contracted,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::SpatialChristoffelSecondKindContracted<
                        DataType, 3, Frame> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_ricci,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SpatialRicci<DataType, 3, Frame> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> weyl_electric_scalar,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::WeylElectricScalar<DataType> /*meta*/) const;

    void operator()(const gsl::not_null<Scalar<DataType>*> weyl_magnetic_scalar,
                    const gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::WeylMagneticScalar<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<Scalar<DataType>*> sqrt_det_spatial_metric,
        const gsl::not_null<CachedBuffer*> cache,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) const;

    void operator()(
        const gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_lapse,
        const gsl::not_null<CachedBuffer*> cache,
        DerivLapse<DataType, Frame> /*meta*/) const;

   private:
    const KerrSchild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
    const std::optional<std::reference_wrapper<const Mesh<3>>>& mesh_;
    const std::optional<std::reference_wrapper<const InverseJacobian<
        DataType, 3, ::Frame::ElementLogical, ::Frame::Inertial>>>&
        inv_jacobian_;
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = KerrSchild::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/);

    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::DerivDetSpatialMetric<DataType, 3, Frame> /*meta*/);

    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/);

    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3,
                                                    Frame> /*meta*/);

    Scalar<DataVector> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::FixedSource<::CurvedScalarWave::Tags::Psi> /*meta*/);

   private:
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

 private:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> dimensionless_spin_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, 3> center_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, volume_dim> boost_velocity_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
  bool zero_spin_{};
  bool zero_velocity_{};
};

SPECTRE_ALWAYS_INLINE bool operator==(const KerrSchild& lhs,
                                      const KerrSchild& rhs) {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center() and
         lhs.boost_velocity() == rhs.boost_velocity();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const KerrSchild& lhs,
                                      const KerrSchild& rhs) {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace Cowling
