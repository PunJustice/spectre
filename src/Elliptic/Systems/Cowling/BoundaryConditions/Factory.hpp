// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/BoundaryConditions/AnalyticSolution.hpp"
#include "Elliptic/Systems/Cowling/BoundaryConditions/Robin.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling::BoundaryConditions {

template <typename System>
using standard_boundary_conditions =
    tmpl::list<Robin<System::volume_dim>>;

}