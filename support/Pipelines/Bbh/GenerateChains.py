# Distributed under the MIT License.
# See LICENSE.txt for details.

############ SKETCH

# - Setup chain parameters.
# - Setup directory structure for ID solves.
#       One Xcts directory and multiple for different couplings?
#       Or one Xcts solve for each coupling?
# - Send with generate-id and schedule=False (except for the first one).
# - Correct submit script to send pure python job.

# Chain parameter options
# - Distance sequence. Forward directly to omegaAndAdot
# - Number of orbits sequence. Determine omega then distance, then
#   omega adot again in InitialOrbitalParameters.py
# - Frequency sequence. Pass orbital frequency and
#   InitialOrbitalParameters.py will find the right separation
# - (Optional) Time to merger sequence. Similar to number of orbits.
# - Couplings sequence. Fixed XCTS background, multiple scalar solves.

# Directory structure (frequency sequence)
# Chain/
#   Omega01/
#       Xcts/
#       ScalarSolve/
#   Omega02/
#       ...
#
# Directory structure (couplings sequence)
# (
# Label by integers not couplings.
#  e.g. ['01', [eta: 1.0, zeta: -1.0], '02', [eta: 2.0, zeta: -20.0], ...]
# )
# Chain/
#   Xcts/
#   ScalarSolve01/
#   ScalarSolve02/
#   ...
# Combined directory structure
# Chain/
#   Omega01/
#       Xcts/
#       ScalarSolve01/
#       ScalarSolve02/
#       ...
#   Omega02/
#       ...
# Append to summary table Chain/Summary.dat after every solve.
# [xcts params, couplings, masses, spins, scalar at horizons]
#
