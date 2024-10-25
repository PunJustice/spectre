# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarGaussBonnetSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ScalarizedGr.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ScalarizedGr.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Simd
  RootFinding
  PUBLIC
  CurvedScalarWave
  DataStructures
  ElasticityPointwiseFunctions
  ErrorHandling
  GeneralRelativity
  GeneralRelativitySolutions
  GrMhdAnalyticData
  InitialDataUtilities
  Options
  RelativisticEulerSolutions
  Serialization
  Utilities
  Xcts
  XctsAnalyticData
  XctsAnalyticSolutions
  XctsPointwiseFunctions
  )
