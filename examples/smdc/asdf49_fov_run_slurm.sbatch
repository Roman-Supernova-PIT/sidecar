#!/bin/bash
#SBATCH --partition mem-med
#SBATCH --time 02:00:00
#SBATCH --output asdf49_fov_run.out

ssh localhost "cd ${HOME}/Roman/pipeline; bash environment/smdc-apptainer-mwv-dev.sh /home/sidecar/examples/smdc/asdf49_fov_run.sh"
