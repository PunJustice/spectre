// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticSolutions/Cowling/Inverser.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/KerrSchildCowling.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Cowling/Zero.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling::Solutions {
using all_analytic_solutions = tmpl::append<tmpl::list<KerrSchild>>;
template <size_t Dim>
using all_initial_guesses = tmpl::append<tmpl::list<Zero<Dim>, Inverser<Dim>>>;
}  // namespace Cowling::Solutions
