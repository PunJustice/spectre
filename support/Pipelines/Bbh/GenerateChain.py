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
from spectre.Pipelines.Bbh.PostprocessId import (
    postprocess_id,
    postprocess_st_id,
)
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Pipelines.Bbh.SolveST import prepare_scalar_solve

logger = logging.getLogger(__name__)


def generate_chain(
    mass_ratio: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    # Orbital parameters
    xcts_executable_path: Union[str, Path],
    scalar_executable_path: Union[str, Path],
    # Scalar tensor parameters
    coupling_sequence: Sequence[tuple[float, float]],
    # coupling_linear: float = 0.0,
    # coupling_quadratic: float = 0.0,
    # coupling_quartic: float = 0.0,
    spectre_home_dir: Optional[Union[str, Path]],
    separation_sequence: Optional[Sequence[float]] = None,
    orbital_angular_velocity_sequence: Optional[Sequence[float]] = None,
    id_parity: bool = True,
    # Resolution
    refinement_level: int = 1,
    polynomial_order: int = 6,
    # Scheduling options
    control: bool = True,
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

    # We might need a new template since things are chained by 'Next'
    ID_INPUT_FILE_NAME = "InitialDataChain.yaml"
    ID_POSTPROC_INPUT_FILE_NAME = "InitialData.yaml"
    SCALAR_ID_INPUT_FILE_NAME = "SolveSTChain.yaml"
    ID_INPUT_FILE_TEMPLATE = (
        f"{spectre_home_dir}/support/Pipelines/Bbh/{ID_INPUT_FILE_NAME}"
    )
    ID_POSTPROC_INPUT_FILE_TEMPLATE = f"{spectre_home_dir}/support/Pipelines/Bbh/{ID_POSTPROC_INPUT_FILE_NAME}"
    SCALAR_ID_INPUT_FILE_TEMPLATE = (
        f"{spectre_home_dir}/support/Pipelines/Bbh/{SCALAR_ID_INPUT_FILE_NAME}"
    )

    # Resolve directories
    if chain_dir:
        chain_dir = Path(chain_dir).resolve()

    initial_orbital_parameters_sequence = np.empty((0, 3), float)

    # Specify only consistent Xcts sequences
    if separation_sequence is not None:
        assert (
            orbital_angular_velocity_sequence is None
        ), "Specify only one sequence."

        # Set up initial orbital parameters and store them in a vector
        for separation in separation_sequence:
            orbital_angular_velocity = None
            radial_expansion_velocity = None
            eccentricity = 0.0
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

            initial_orbital_parameters_sequence = np.append(
                initial_orbital_parameters_sequence,
                [
                    [
                        separation,
                        orbital_angular_velocity,
                        radial_expansion_velocity,
                    ]
                ],
                axis=0,
            )

    if orbital_angular_velocity_sequence is not None:
        assert separation_sequence is None, "Specify only one sequence."
        for orbital_angular_velocity in orbital_angular_velocity_sequence:
            separation = None
            radial_expansion_velocity = None
            eccentricity = 0.0
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

            initial_orbital_parameters_sequence = np.append(
                initial_orbital_parameters_sequence,
                [
                    [
                        separation,
                        orbital_angular_velocity,
                        radial_expansion_velocity,
                    ]
                ],
                axis=0,
            )

    # Print ID orbital paramters
    logger.info(
        "Initial data parameters:"
        f" {pretty_repr(initial_orbital_parameters_sequence)}"
    )

    # File to write summary of parameters and horizon quantities after
    # each scalar solve
    summary_data_file = open(f"{chain_dir}/ChainSummary.txt", "w")

    iteration = 0

    for (
        initial_orbital_parameters_element
    ) in initial_orbital_parameters_sequence:
        separation = initial_orbital_parameters_element[0]
        orbital_angular_velocity = initial_orbital_parameters_element[1]
        radial_expansion_velocity = initial_orbital_parameters_element[2]
        # Set up Xcts directory
        xcts_run_dir = f"{chain_dir}/Xcts{iteration:02}"

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
            # coupling_linear=coupling_linear,
            # coupling_quadratic=coupling_quadratic,
            # coupling_quartic=coupling_quartic,
            # id_parity=id_parity,
            # Use the template here. Exclude Next
            id_input_file_template=ID_INPUT_FILE_TEMPLATE,
            # Exec
            executable=xcts_executable_path,
            # Specify run directory
            run_dir=xcts_run_dir,
            # This should be kept disabled
            control=False,
            validate=False,
            # Always disabled
            evolve=False,
            # Already submitted pure python job
            scheduler=None,
            refinement_level=refinement_level,
            polynomial_order=polynomial_order,
        )

        # Postprocess. Find Horizons and do control loop
        # Note: The control loop uses Next to compute the horizons.
        # It also uses the InitialData.yaml template -- since the
        # control_id routine does not take input file templates.
        # We also need to pass the executable path since for some
        # reason is not retreiveing it when we use the pipeline
        # functions outside without ./spectre bbh ...
        #
        # How to improve this?
        # One option to make this less confusing is to write a
        # postprocess function that only computes the horizon,
        # and add the option of passing an input file template
        # to control_id
        last_control_id_input_file_path = postprocess_id(
            id_input_file_path=f"{xcts_run_dir}/{ID_INPUT_FILE_NAME}",
            id_run_dir=None,
            horizon_l_max=12,
            horizons_file=None,
            # Control loop option
            control=control,
            control_refinement_level=refinement_level,
            control_polynomial_order=polynomial_order,
            # dimensionless_coupling_linear=coupling_linear,
            # dimensionless_coupling_quadratic=coupling_quadratic,
            # dimensionless_coupling_quartic=coupling_quartic,
            # initial_guess_same_parity=id_parity,
            # Use the template here. Exclude Next
            id_input_file_template=ID_POSTPROC_INPUT_FILE_TEMPLATE,
            executable=xcts_executable_path,
            validate=False,
            # Check this argument
            scalar_solve=False,
            evolve=False,
            pipeline_dir=None,
            scheduler=None,
        )

        # Scalar solve
        if scalar_solve:
            scalar_iteration = 0
            for coupling_quadratic, coupling_quartic in coupling_sequence:
                # Scalar run dir
                scalar_run_dir = (
                    f"{xcts_run_dir}/ScalarSolve{scalar_iteration:02}"
                )

                prepare_scalar_solve(
                    # Need to know the path to the last control solve
                    # id_input_file_path=f"{xcts_run_dir}/{ID_INPUT_FILE_NAME}",
                    id_input_file_path=last_control_id_input_file_path,
                    dimensionless_coupling_linear=0.0,
                    dimensionless_coupling_quadratic=coupling_quadratic,
                    dimensionless_coupling_quartic=coupling_quartic,
                    initial_guess_same_parity=id_parity,
                    id_run_dir=None,
                    id_input_file_template=SCALAR_ID_INPUT_FILE_TEMPLATE,
                    run_dir=scalar_run_dir,
                    pipeline_dir=None,
                    refinement_level=refinement_level,
                    polynomial_order=polynomial_order,
                    executable=scalar_executable_path,
                    scheduler=None,
                )

                # Postprocess. Find Horizons and do control loop
                # Note: For the control loop we do not need to make use of Next.
                final_horizon_values = postprocess_st_id(
                    id_input_file_path=f"{scalar_run_dir}/SolveSTChain.yaml",
                    id_run_dir=None,
                    horizon_l_max=12,
                    horizons_file=None,
                    # Control loop off for now.
                    # Use the template here. Exclude Next
                    id_input_file_template=ID_INPUT_FILE_TEMPLATE,
                    # Check this argument
                    evolve=False,
                    scheduler=None,
                )

                final_horizon_values.update(
                    {
                        "Separation": separation,
                        "OrbitalAngularVelocity": orbital_angular_velocity,
                        "RadialExpansionVelocity": radial_expansion_velocity,
                        "CouplingQuadratic": coupling_quadratic,
                        "CouplingQuartic": coupling_quartic,
                        "IdParity": int(id_parity),
                    }
                )

                logger.info(
                    "Initial data parameters:"
                    f" {pretty_repr(final_horizon_values)}"
                )

                # We omit the spin direction
                # (Include resolution as well?)
                summary_data_file.write(
                    "{Separation}, {OrbitalAngularVelocity},"
                    " {RadialExpansionVelocity}, {CouplingQuadratic},"
                    " {CouplingQuartic}, {IdParity}, {AreaAhA},"
                    " {IrreducibleMassAhA}, {ChristodoulouMassAhA},"
                    " {DimensionlessSpinMagnitudeAhA},"
                    " {SurfaceAverageOfScalarAhA}, {AreaAhB},"
                    " {IrreducibleMassAhB}, {ChristodoulouMassAhB},"
                    " {DimensionlessSpinMagnitudeAhB},"
                    " {SurfaceAverageOfScalarAhB} \n".format(
                        **final_horizon_values
                    )
                )
                summary_data_file.flush()

                scalar_iteration += 1

        iteration += 1

    return


if __name__ == "__main__":
    # Submit this script with support/SubmitScripts/SubmitChain.sh
    # and adapt it until we automatize that

    # Compile cli, SolveXcts and SolveCowling before running
    # Setup the PYTHONPATH in the submit script before running

    # Set to DEBUG for more output
    logging.basicConfig(level=logging.INFO)

    # Need to specify the build and chain directories
    spectre_home_dir = "/u/guilara/repos/others_spectre/PJSpectre/spectre"
    build_dir = f"{spectre_home_dir}/build_pip_st"
    chain_dir = "/urania/ptmp/guilara/spectre/Elliptic/Binary/2024/STTests/Pipeline/TestID/ChainDir"

    # Notes:
    # - If you change the yaml files make sure the XYZChain.yaml and the XYZ.yaml
    #   version are identical except for the 'Next' arguments.
    # - Adjust the number of nodes in the submit script.
    # - Use distances << 60 M (where the envelope fixed radius is)
    #   until we automatically scale it (linearly with separation?).
    # - Send different parities in different jobs to avoid confusion and sorting.
    # - Use only one sequence for the orbital parameters (separation or orbital
    #   frequency).
    # - Xcts solutions are stored in XctsXX/ folders. Scalar solutions using
    #   these Xcts backgrounds are stored in subfolders ScalarSolveXX.
    # - A summary of the relevant run parameters and horizon values is output on
    #   chain_dir/ChainSummary.txt. See above for the order of these parameters.
    #   We can use this to plot e.g. the avg scalar at horizon vs frequency.
    #   (Shall we add as well the resolution data?)
    # - I have not tested the linear coupling, so I don't know if its well
    #   behaved

    generate_chain(
        spectre_home_dir=spectre_home_dir,
        mass_ratio=2.0,
        dimensionless_spin_a=[0.0, 0.0, 0.0],
        dimensionless_spin_b=[0.0, 0.0, 0.0],
        # Orbital parameter sequences
        separation_sequence=[16.0, 32.0],
        # orbital_angular_velocity_sequence=[0.016, 0.017, 0.018],
        coupling_sequence=[[2.0, -20.0]],
        # Control
        control=True,
        # Parity. True for like charges. False for opposite.
        id_parity=False,
        # Resolution
        refinement_level=1,
        polynomial_order=5,
        # Scheduling options
        chain_dir=chain_dir,
        xcts_executable_path=f"{build_dir}/bin/SolveXcts",
        scalar_solve=True,
        scalar_executable_path=f"{build_dir}/bin/SolveCowling",
    )
