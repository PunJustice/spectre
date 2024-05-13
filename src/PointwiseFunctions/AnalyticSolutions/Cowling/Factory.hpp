// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Inverser.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/SuperposedInverser.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Zero.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling::Solutions {
using all_analytic_solutions = tmpl::list<
    Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>,
    Xcts::Solutions::WrappedGr<gr::Solutions::HarmonicSchwarzschild>>;
template <size_t Dim>
using all_initial_guesses =
    tmpl::list<Zero<Dim>, Inverser<Dim>, SuperposedInverser<Dim>>;
}  // namespace Cowling::Solutions
