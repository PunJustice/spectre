// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Elliptic/Systems/Cowling/BoundaryConditions/DoNothing.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Utilities/TMPL.hpp"

namespace Cowling::BoundaryConditions {

using standard_boundary_conditions =
    tmpl::list<::Poisson::BoundaryConditions::Robin<3>, DoNothing>;

}
