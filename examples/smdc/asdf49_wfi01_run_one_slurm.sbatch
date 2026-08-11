#!/bin/bash
#SBATCH --partition mem-med
#SBATCH --time 00:30:00
#SBATCH --output asdf49_wfi01_run_one.out

ssh localhost "cd ${HOME}/snpit/packages; bash environment/smdc-apptainer-mwv-dev.sh /home/sidecar/examples/smdc/asdf49_wfi01_run_one.sh"
