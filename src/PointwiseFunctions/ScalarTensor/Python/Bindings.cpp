// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarCharge.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace py = pybind11;

namespace ScalarTensor::py_bindings {

void bind_st_charge_impl(py::module& m) {
  m.def(
      "st_horizon_quantities",
      [](ylm::Strahlkorper<Frame::Inertial> sphere,
         tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric,
         tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric,
         tnsr::i<DataVector, 3, Frame::Inertial> phi) -> py::dict {
        const auto box = db::create<
            tmpl::list<
                ylm::Tags::Strahlkorper<Frame::Inertial>,
                gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
                gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
                CurvedScalarWave::Tags::Phi<3>>,
            tmpl::list<
                ylm::Tags::ThetaPhiCompute<::Frame::Inertial>,
                ylm::Tags::RadiusCompute<::Frame::Inertial>,
                ylm::Tags::RhatCompute<::Frame::Inertial>,
                ylm::Tags::InvJacobianCompute<::Frame::Inertial>,
                ylm::Tags::JacobianCompute<::Frame::Inertial>,
                ylm::Tags::DxRadiusCompute<::Frame::Inertial>,
                ylm::Tags::NormalOneFormCompute<::Frame::Inertial>,
                ylm::Tags::OneOverOneFormMagnitudeCompute<DataVector, 3,
                                                          ::Frame::Inertial>,
                ylm::Tags::UnitNormalOneFormCompute<::Frame::Inertial>,
                ylm::Tags::UnitNormalVectorCompute<::Frame::Inertial>,
                gr::surfaces::Tags::AreaElementCompute<::Frame::Inertial>,
                ScalarTensor::StrahlkorperScalar::Tags::
                    ScalarChargeIntegrandCompute,
                gr::surfaces::Tags::SurfaceIntegralCompute<
                    ScalarTensor::StrahlkorperScalar::Tags::
                        ScalarChargeIntegrand,
                    ::Frame::Inertial>>>(
            std::move(sphere), std::move(spatial_metric),
            std::move(inv_spatial_metric), std::move(phi));
        py::dict result{};
        result["SurfaceAverageOfScalar"] =
            db::get<gr::surfaces::Tags::SurfaceIntegral<
                ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrand,
                ::Frame::Inertial>>(box);
        return result;
      },
      py::arg("sphere"), py::arg("spatial_metric"),
      py::arg("inv_spatial_metric"), py::arg("phi"));
}

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.SphericalHarmonics");
  bind_st_charge_impl(m);
}

}  // namespace ScalarTensor::py_bindings
