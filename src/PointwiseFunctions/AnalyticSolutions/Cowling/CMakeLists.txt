# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Inverser.hpp
  SuperposedInverser.hpp
  Zero.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  DataStructures
  ErrorHandling
  MathFunctions
  Options
  Cowling
  Hydro
  Serialization
  Interpolation
  Spectral
  Utilities
  GeneralRelativity
  InitialDataUtilities
  Xcts
  XctsAnalyticData
  XctsSolutions
  XctsPointwiseFunctions
  ElasticityPointwiseFunctions
  GeneralRelativity
  GeneralRelativitySolutions
  )
