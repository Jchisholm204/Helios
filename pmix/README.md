# PMIx Test Platform
(build system taken from [AURORA](https://github.com/Jchisholm204/AURORA))

## Instructions
- PMIx programs must be launched with an active daemon.
    - Most clusters use Slurm.
    - Running the `prte` daemon beforehand also works.
    - MPI launches the `prte` daemon automatically with `mpirun`
