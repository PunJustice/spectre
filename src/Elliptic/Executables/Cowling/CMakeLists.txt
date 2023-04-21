# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SolveCowling)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveCowling.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
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
  CowlingSolutions
  Utilities
  )

