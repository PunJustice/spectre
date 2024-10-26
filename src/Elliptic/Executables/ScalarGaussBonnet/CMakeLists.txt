# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SolveScalarGaussBonnet)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SolveScalarGaussBonnet.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  AmrCriteria
  Charmxx::main
  Convergence
  CoordinateMaps
  DataStructures
  DiscontinuousGalerkin
  Domain
  DomainCreators
  Elliptic
  EllipticDg
  EllipticDgSubdomainOperator
  EllipticSubdomainPreconditioners
  ErrorHandling
  Events
  EventsAndTriggers
  FunctionsOfTime
  Hydro
  GrMhdAnalyticData
  Informer
  InitialDataUtilities
  Initialization
  LinearOperators
  MathFunctions
  Observer
  Options
  Parallel
  ParallelAmr
  ParallelLinearSolver
  ParallelMultigrid
  ParallelNonlinearSolver
  ParallelSchwarz
  PhaseControl
  Utilities
  ScalarGaussBonnet
  ScalarGaussBonnetSolutions
  ScalarGaussBonnetBoundaryConditions
  Xcts
  XctsAnalyticData
  XctsEvents
  XctsSolutions
  )
