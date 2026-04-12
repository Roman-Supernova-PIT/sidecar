#!/bin/bash
#SBATCH --account m4385
#SBATCH --constraint gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 60G
#SBATCH --gpus-per-task 1
#SBATCH --qos debug
#SBATCH --time 00:10:00
#SBATCH --output asdf49_wfi01_run_one.out

export WHICHROMANENV=cuda-dev
bash interactive-podman-rknop-dev.sh /home/sidecar/examples/perlmutter/asdf49_wfi01_run_one.sh
