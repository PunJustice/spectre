# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY CowlingSolutions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  KerrSchildCowling.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Inverser.hpp
  SuperposedInverser.hpp
  KerrSchildCowling.hpp
  Zero.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  MathFunctions
  Options
  Cowling
  Serialization
  SpecialRelativity
  Interpolation
  Utilities
  GeneralRelativity
  )
