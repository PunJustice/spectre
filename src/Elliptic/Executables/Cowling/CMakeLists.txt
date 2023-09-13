# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Convergence
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  Events
  EventsAndTriggers
  Importers
  Informer
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelLinearSolver
  ParallelMultigrid
  ParallelSchwarz
  Cowling
  CowlingBoundaryConditions
  CowlingSolutions
  Utilities
  )

function(add_cowling_executable DIM)
  add_spectre_parallel_executable(
    "SolveCowling${DIM}D"
    SolveCowling
    Elliptic/Executables/Cowling
    "Metavariables<${DIM}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_cowling_executable)

add_Cowling_executable(1)
add_Cowling_executable(2)
add_Cowling_executable(3)
