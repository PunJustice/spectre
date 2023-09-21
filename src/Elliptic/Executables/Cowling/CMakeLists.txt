# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
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
  Xcts
  XctsAnalyticData
  XctsSolutions
  GeneralRelativity
  GeneralRelativitySolutions
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

