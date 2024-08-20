#!/bin/bash -
#SBATCH -o /urania/ptmp/guilara/spectre/Elliptic/Binary/2024/STTests/Pipeline/TestID/ChainDir/spectre.out
#SBATCH -e /urania/ptmp/guilara/spectre/Elliptic/Binary/2024/STTests/Pipeline/TestID/ChainDir/spectre.out
#SBATCH -J testpip
#SBATCH --no-requeue
#SBATCH --comment "SPECTRE_INPUT_FILE=InitialData.yaml"
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH -t 03:59:00
#SBATCH -p p.debug

# Distributed under the MIT License.
# See LICENSE.txt for details.

# This script is a template for submitting jobs to a cluster with Slurm.
# See `support/Python/Schedule.py` for how this template is used and how
# placeholders are resolved.

export RUN_DIR=/urania/ptmp/guilara/spectre/Elliptic/Binary/2024/STTests/Pipeline/TestID/ChainDir
export SPECTRE_INPUT_FILE=InitialData.yaml
export SPECTRE_EXECUTABLE=/urania/u/guilara/repos/spectre/build_bbh_jul2024/bin/SolveXcts
export SPECTRE_CHECKPOINT=
export SPECTRE_CLI=/u/guilara/repos/others_spectre/PJSpectre/spectre/build_pip_st/bin/spectre

# One thread for communication
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 2)

echo "###################################"
echo "######       JOB INFO        ######"
echo "###################################"
echo
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Directory: ${RUN_DIR}"
echo "Submit Directory: ${SLURM_SUBMIT_DIR}"
echo "Queue: ${SLURM_JOB_PARTITION}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK}"
echo "Charm ppn: ${CHARM_PPN}"
echo "PATH: ${PATH}"
echo "Executable: ${SPECTRE_EXECUTABLE}"
echo "CLI: ${SPECTRE_CLI}"
echo

# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
module purge
module load gcc/11
module load impi/2021.7
module load boost/1.79
module load gsl/1.16
module load cmake/3.26
module load hdf5-serial/1.12.2
module load anaconda/3/2021.11

# Load Spack environment
source /u/guilara/repos/spack/share/spack/setup-env.sh
spack env activate env3_spectre_impi

# Define Charm paths
export CHARM_ROOT=/u/guilara/charm_impi_2/mpi-linux-x86_64-smp
export PATH=$PATH:/u/guilara/charm_impi_2/mpi-linux-x86_64-smp/bin

# Load python environment
export SPECTRE_HOME=/u/guilara/repos/others_spectre/PJSpectre/spectre
source $SPECTRE_HOME/env/bin/activate
source $SPECTRE_HOME/build_pip_st/bin/LoadPython.sh
export LD_PRELOAD=/u/guilara/repos/spack/var/spack/environments/env3_spectre_impi/.spack-env/view/lib/libjemalloc.so
############################################################################
# Set desired permissions for files created with this script
umask 0022

echo
echo "###################################"
echo "######   Executable Output   ######"
echo "###################################"
echo

cd ${RUN_DIR}

python3 /u/guilara/repos/others_spectre/PJSpectre/spectre/support/Pipelines/Bbh/GenerateChain.py
exit_code=$?
exit $exit_code

# srun -n ${SLURM_NTASKS} ${SPECTRE_EXECUTABLE} \
#     --input-file ${SPECTRE_INPUT_FILE} \
#     ++ppn ${CHARM_PPN} +pemap 0-34,36-70 +commap 35,71 \
#     ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}

# exit_code=$?


# # Run next entrypoint listed in input file
# if [ $exit_code -eq 0 ]; then
#   sleep 10s
#   ${SPECTRE_CLI} run-next ${SPECTRE_INPUT_FILE} -i .
#   exit $?
# fi

# exit $exit_code
