// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Inverser.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/SuperposedInverser.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Zero.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/KerrSchildCowling.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling::Solutions {
using all_analytic_solutions =
    tmpl::list<KerrSchild>;
template <size_t Dim>
using all_initial_guesses =
    tmpl::list<Zero<Dim>, Inverser<Dim>, SuperposedInverser<Dim>>;
}  // namespace Cowling::Solutions
