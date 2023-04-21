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

add_spectre_parallel_executable(
  SolveCowling
  SolveCowling
  Elliptic/Executables/Cowling
  "Metavariables"
  "${LIBS_TO_LINK}"
  )
