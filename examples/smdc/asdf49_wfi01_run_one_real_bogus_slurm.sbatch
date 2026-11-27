#!/bin/bash
#SBATCH --partition mem-med
#SBATCH --time 00:30:00
#SBATCH --output asdf49_wfi01_run_one_real_bogus.out

# Run examples/smdc/fetch_real_bogus_models.sh once before submitting this.
ssh localhost "cd ${HOME}/snpit/packages; bash environment/smdc-apptainer-mwv-dev.sh /home/sidecar/examples/smdc/asdf49_wfi01_run_one_real_bogus.sh"
