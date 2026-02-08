#!/bin/bash
#SBATCH --account m4385
#SBATCH --constraint gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
###SBATCH --mem 60G
#SBATCH --gpus-per-task 1
#SBATCH --qos debug
#SBATCH --time 00:30:00
#SBATCH --output sidecar_demo_1_subtraction.out


# This script should be submitted the directory that will become /home
# I.e. sbatch sidecar/examples/perlmutter/nov_R_images_600_630_slurm.sh
ROMAN_DIR=$PWD

podman-hpc run --gpu \
    --mount type=bind,source=$ROMAN_DIR,target=/home \
    --mount type=bind,source=$HOME/secrets,target=/secrets \
    --mount type=bind,source=$PSCRATCH/snpit_temp,target=/snpit_temp \
    --mount type=bind,source=$ROMAN_DIR/sn_info_dir,target=/sn_info_dir \
    --mount type=bind,source=$ROMAN_DIR/dia_out_dir,target=/dia_out_dir \
    --mount type=bind,source=$ROMAN_DIR/lc_out_dir,target=/lc_out_dir \
    --mount type=bind,source=/pscratch/sd/m/masao/roman_snpit/database_dirs,target=/data \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data,target=/ou2024 \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data,target=/sims_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE,target=/ou2024_snana \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/ROMAN+LSST_LARGE_SNIa-normal,target=/ou2024_snana_lc_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/sims_sed_library,target=/ou2024_sims_sed_library \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/m4385/calib_data/A25ePSF,target=/a25epsf \
    --env LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
    --env PYTHONPATH=/roman_imsim:/phrosty:/sfft \
    --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --env OMP_NUM_THREADS=1 \
    --env VECLIB_MAXIMUM_THREADS=1 \
    --env SIMS_DIR=/sims_dir \
    --env SNANA_PQ_DIR=/snana_pq_dir \
    --env SN_INFO_DIR=/sn_info_dir \
    --env DIA_OUT_DIR=/dia_out_dir \
    --env TERM=xterm \
    --env SNPIT_CONFIG=/home/sidecar/config/sidecar_config.yaml \
    --annotation run.oci.keep_original_groups=1 \
    -w /home \
    -it \
    registry.nersc.gov/m4385/rknop/roman-snpit-env:cuda-dev \
    bash -c " \
    pip install -e sidecar; \
    python sidecar/sidecar/pipeline.py \
	--image-collection snpitdb --image-provenance-tag asdf_functional_test --image-process load_rdm_image \
        --science-observation-id 99999010010010010010030 \
        --science-band F062 \
        --science-sca 2 \
    "

# Once `sidecar` is part of container image, the 'pip install -e sidecar' will be removed.
