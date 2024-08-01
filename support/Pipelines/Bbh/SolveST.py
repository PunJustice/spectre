# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

import spectre.IO.H5 as spectre_h5
from spectre.Pipelines.Bbh.InitialData import id_parameters
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

SCALARSOLVE_INPUT_FILE_TEMPLATE = Path(__file__).parent / "SolveST.yaml"


def prepare_scalar_solve(
    id_input_file_path: Union[str, Path],
    dimensionless_coupling_linear: float,
    dimensionless_coupling_quadratic: float,
    dimensionless_coupling_quartic: float,
    id_run_dir: Optional[Union[str, Path]] = None,
    refinement_level: int = 1,
    polynomial_order: int = 6,
    **scheduler_kwargs,
):
    """Set up scalar solve parameters form Xcts ID.

    Arguments:
    """

    # Read input file
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    binary_data = id_input_file["Background"]["Binary"]
    M_input_A = binary_data["ObjectRight"]["KerrSchild"]["Mass"]
    M_input_B = binary_data["ObjectLeft"]["KerrSchild"]["Mass"]
    chi_input_A = binary_data["ObjectRight"]["KerrSchild"]["Spin"]
    chi_input_B = binary_data["ObjectLeft"]["KerrSchild"]["Spin"]
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    orbital_angular_velocity = binary_data["AngularVelocity"]
    radial_expansion_velocity = binary_data["Expansion"]

    # Get black hole physical parameters
    with spectre_h5.H5File(f"{id_run_dir}/Horizons.h5", "r") as horizons_file:
        AhA_quantities = to_dataframe(horizons_file.get_dat("AhA.dat")).iloc[-1]

        M_A = AhA_quantities["ChristodoulouMass"]
        chi_A_x = AhA_quantities["DimensionlessSpinVector_x"]
        chi_A_y = AhA_quantities["DimensionlessSpinVector_y"]
        chi_A_z = AhA_quantities["DimensionlessSpinVector_z"]

        horizons_file.close_current_object()
        AhB_quantities = to_dataframe(horizons_file.get_dat("AhB.dat")).iloc[-1]

        M_B = AhB_quantities["ChristodoulouMass"]
        chi_B_x = AhB_quantities["DimensionlessSpinVector_x"]
        chi_B_y = AhB_quantities["DimensionlessSpinVector_y"]
        chi_B_z = AhB_quantities["DimensionlessSpinVector_z"]

    # Compute dimensionfull couplings with respect to Christodoulou masses
    coupling_linear_M_A = dimensionless_coupling_linear * (M_A**2)
    coupling_quadratic_M_A = dimensionless_coupling_quadratic * (M_A**2)
    coupling_quartic_M_A = dimensionless_coupling_quartic * (M_A**2)

    coupling_linear_M_B = dimensionless_coupling_linear * (M_B**2)
    coupling_quadratic_M_B = dimensionless_coupling_quadratic * (M_B**2)
    coupling_quartic_M_B = dimensionless_coupling_quartic * (M_B**2)

    # Initial guesses
    if coupling_quadratic_M_A > 1e-16:
        initial_guess_amplitude_M_A = np.sqrt(
            np.abs(coupling_quartic_M_A / coupling_quadratic_M_A)
        )
        initial_guess_amplitude_M_B = -np.sqrt(
            np.abs(coupling_quartic_M_A / coupling_quadratic_M_A)
        )
    else:
        initial_guess_amplitude_M_A = 0.3
        initial_guess_amplitude_M_B = -0.3

    # Roll-off location
    roll_off_location = 0.93 / orbital_angular_velocity

    # Run ID
    generate_scalar_tensor_id(
        mass_a=M_input_A,
        mass_b=M_input_B,
        dimensionless_spin_a=chi_input_A,
        dimensionless_spin_b=chi_input_B,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        coupling_linear=coupling_linear_M_A,
        coupling_quadratic=coupling_quadratic_M_A,
        coupling_quartic=coupling_quartic_M_A,
        roll_off_location=roll_off_location,
        initial_guess_amplitude_a=initial_guess_amplitude_M_A,
        initial_guess_amplitude_b=initial_guess_amplitude_M_B,
        control=False,
        evolve=False,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
        **scheduler_kwargs,
    )

    return 0.0


def generate_scalar_tensor_id(
    mass_a: float,
    mass_b: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    # Orbital parameters
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    # Scalar tensor parameters
    coupling_linear: float,
    coupling_quadratic: float,
    coupling_quartic: float,
    roll_off_location: float,
    initial_guess_amplitude_a: float,
    initial_guess_amplitude_b: float,
    # Resolution
    refinement_level: int = 1,
    polynomial_order: int = 6,
    # Scheduling options
    id_input_file_template: Union[str, Path] = SCALARSOLVE_INPUT_FILE_TEMPLATE,
    control: bool = False,
    evolve: bool = False,
    run_dir: Optional[Union[str, Path]] = None,
    pipeline_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Generate Scalar Tensor initial data for a BBH simulation.

    Parameters for the initial data will be inserted into the
    'id_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    The orbital parameters can be computed with the function
    'initial_orbital_parameters' in
    'support.Pipelines.EccentricityControl.InitialOrbitalParameters'.

    Intrinsic parameters:
      mass_a: Mass of the larger black hole.
      mass_b: Mass of the smaller black hole.
      dimensionless_spin_a: Dimensionless spin of the larger black hole, chi_A.
      dimensionless_spin_b: Dimensionless spin of the smaller black hole, chi_B.

    Orbital parameters:
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.

    Scalar tensor parameters parameters:
      coupling linear: float,
      coupling quadratic: float,
      coupling quartic: float.
      roll_off_location: float,
      initial_guess_amplitude_a: float,
      initial_guess_amplitude_b: float,

    Scheduling options:
      id_input_file_template: Input file template where parameters are inserted.
      control: If set to True, a postprocessing control loop will adjust the
        input parameters to drive the horizon masses and spins to the specified
        values. If set to False, the horizon masses and spins in the generated
        data will differ from the input parameters. (default: False)
      evolve: Set to True to evolve the initial data after generation.
      pipeline_dir: Directory where steps in the pipeline are created. Required
        when 'evolve' is set to True. The initial data will be created in a
        subdirectory '001_InitialData'.
      run_dir: Directory where the initial data is generated. Mutually exclusive
        with 'pipeline_dir'.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    assert segments_dir is None, (
        "Initial data generation doesn't use segments at the moment. Specify"
        " '--run-dir' / '-o' or '--pipeline-dir' / '-d' instead."
    )
    if evolve:
        assert pipeline_dir is not None, (
            "Specify a '--pipeline-dir' / '-d' to evolve the initial data."
            " Don't specify a '--run-dir' / '-o' because it will be created in"
            " the 'pipeline_dir' automatically."
        )
        assert run_dir is None, (
            "Specify the '--pipeline-dir' / '-d' rather than '--run-dir' / '-o'"
            " when evolving the initial data. Directories for the initial data,"
            " evolution, etc will be created in the 'pipeline_dir'"
            " automatically."
        )
    if pipeline_dir and not run_dir:
        run_dir = pipeline_dir / "001x_ScalarTensorInitialData"

    # Determine remaining initial data parameters from options
    id_params = id_parameters(
        mass_a=mass_a,
        mass_b=mass_b,
        dimensionless_spin_a=dimensionless_spin_a,
        dimensionless_spin_b=dimensionless_spin_b,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Initial data parameters: {pretty_repr(id_params)}")

    # Append extra scalar tensor parameters
    id_params.update(
        {
            "Epsilon1": coupling_linear,
            "Epsilon2": coupling_quadratic,
            "Epsilon4": coupling_quartic,
            "RolloffLocation": roll_off_location,
            "InitialGuessAmplitudeA": initial_guess_amplitude_a,
            "InitialGuessAmplitudeB": initial_guess_amplitude_b,
        }
    )
    logger.debug(f"Extended initial data parameters: {pretty_repr(id_params)}")

    # Schedule!
    return schedule(
        id_input_file_template,
        **id_params,
        **scheduler_kwargs,
        control=False,
        evolve=False,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir / "ScalarSolve",
        segments_dir=segments_dir,
    )


@click.command(name="generate-id-st", help=generate_scalar_tensor_id.__doc__)
@click.argument(
    "id_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--dimensionless-coupling-linear",
    type=float,
    help="Dimensionless coupling linear",
    default=0.0,
    show_default=True,
)
@click.option(
    "--dimensionless-coupling-quadratic",
    type=float,
    help="Dimensionless coupling quadratic",
    default=0.0,
    show_default=True,
)
@click.option(
    "--dimensionless-coupling-quartic",
    type=float,
    help="Dimensionless coupling quartic",
    default=0.0,
    show_default=True,
)
@click.option(
    "-i",
    "--id-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the initial data run. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the ID_INPUT_FILE_PATH",
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=1,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=9,
    show_default=True,
)
@scheduler_options
def generate_scalar_tensor_id_command(
    **kwargs,
):
    _rich_traceback_guard = True  # Hide traceback until here
    prepare_scalar_solve(**kwargs)


if __name__ == "__main__":
    generate_scalar_tensor_id_command(help_option_names=["-h", "--help"])
