# Distributed under the MIT License.
# See LICENSE.txt for details.

# Submit and run this python script on the compute node

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np
from rich.pretty import pretty_repr

from spectre.Pipelines.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Pipelines.Bbh.SolveST import prepare_scalar_solve

logger = logging.getLogger(__name__)

# We might need a new template since things are chained by 'Next'
ID_INPUT_FILE_NAME = "InitialDataChain.yaml"
ID_INPUT_FILE_TEMPLATE = Path(__file__).parent / ID_INPUT_FILE_NAME


def generate_chain(
    mass_ratio: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    # Orbital parameters
    separation_sequence: Sequence[float],
    xcts_executable_path: Union[str, Path],
    scalar_executable_path: Union[str, Path],
    # Scalar tensor parameters
    coupling_linear: float = 0.0,
    coupling_quadratic: float = 0.0,
    coupling_quartic: float = 0.0,
    id_parity: bool = True,
    # Resolution
    refinement_level: int = 1,
    polynomial_order: int = 6,
    # Scheduling options
    id_input_file_template: Union[str, Path] = ID_INPUT_FILE_TEMPLATE,
    control: bool = False,
    scalar_solve: bool = False,
    evolve: bool = False,
    chain_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Generate a chain of initial datatsets.

    Arguments:

    Scheduling options:
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Resolve directories
    if chain_dir:
        chain_dir = Path(chain_dir).resolve()

    iteration = 0
    for separation in separation_sequence:
        # Set up Xcts directory
        xcts_run_dir = f"{chain_dir}/Xcts{iteration:02}"

        orbital_angular_velocity = None
        radial_expansion_velocity = None
        eccentricity = 0.
        mean_anomaly_fraction = None
        num_orbits = None
        time_to_merger = None

        # Set up Xcts parameters
        separation, orbital_angular_velocity, radial_expansion_velocity = (
            initial_orbital_parameters(
                mass_ratio=mass_ratio,
                dimensionless_spin_a=dimensionless_spin_a,
                dimensionless_spin_b=dimensionless_spin_b,
                separation=separation,
                orbital_angular_velocity=orbital_angular_velocity,
                radial_expansion_velocity=radial_expansion_velocity,
                eccentricity=eccentricity,
                mean_anomaly_fraction=mean_anomaly_fraction,
                num_orbits=num_orbits,
                time_to_merger=time_to_merger,
            )
        )

        mass_a = mass_ratio / (1.0 + mass_ratio)
        mass_b = 1.0 / (1.0 + mass_ratio)

        generate_id(
            mass_a=mass_a,
            mass_b=mass_b,
            dimensionless_spin_a=dimensionless_spin_a,
            dimensionless_spin_b=dimensionless_spin_b,
            separation=separation,
            orbital_angular_velocity=orbital_angular_velocity,
            radial_expansion_velocity=radial_expansion_velocity,
            coupling_linear=coupling_linear,
            coupling_quadratic=coupling_quadratic,
            coupling_quartic=coupling_quartic,
            id_parity=id_parity,
            # Use the template here. Exclude Next
            id_input_file_template=ID_INPUT_FILE_TEMPLATE,
            # Exec
            executable=xcts_executable_path,
            # Specify run directory
            run_dir=xcts_run_dir,
            # Enable later
            control=True,
            validate=False,
            # Always disabled
            evolve=False,
            # Already submitted pure python job
            scheduler=None,
            refinement_level=refinement_level,
            polynomial_order=polynomial_order,
        )

        if scalar_solve:
            scalar_iteration = 0
            # Scalar run dir
            scalar_run_dir = f"{xcts_run_dir}/ScalarSolve{scalar_iteration:02}"

            prepare_scalar_solve(
                id_input_file_path=f"{xcts_run_dir}/{ID_INPUT_FILE_NAME}",
                dimensionless_coupling_linear=coupling_linear,
                dimensionless_coupling_quadratic=coupling_quadratic,
                dimensionless_coupling_quartic=coupling_quartic,
                initial_guess_same_parity=id_parity,
                id_run_dir=scalar_run_dir,
                pipeline_dir=None,
                refinement_level=refinement_level,
                polynomial_order=polynomial_order,
                executable=scalar_executable_path,
                scheduler=None,
            )

        iteration += 1

    return


if __name__ == "__main__":
    generate_chain(
        mass_ratio=1.0,
        dimensionless_spin_a=[0.0, 0.0, 0.0],
        dimensionless_spin_b=[0.0, 0.0, 0.0],
        # Orbital parameters
        separation_sequence=[14.0, 16.0],
        # Scheduling options
        id_input_file_template=ID_INPUT_FILE_TEMPLATE,
        chain_dir="/urania/ptmp/guilara/spectre/Elliptic/Binary/2024/STTests/Pipeline/TestID/ChainDir",
        xcts_executable_path="/urania/u/guilara/repos/spectre/build_bbh_jul2024/bin/SolveXcts",
        scalar_solve=True,
        scalar_executable_path="/urania/u/guilara/repos/spectre/build_bbh_jul2024/bin/SolveCowling",
    )


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
