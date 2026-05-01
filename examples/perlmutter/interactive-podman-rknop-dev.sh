#!/usr/bin/bash

# 2026-05-01  MWV:
# This should go away by 2027-07-01 when Rob and MWV update the environments successfully on NERSC.
# For now this is stored here for reference.
# To get the bindmounts correct, this file should be executed from the directory above the sidecar checkout.

# See comments in nov2025_container_config.yaml for instructions

export PODMANHPC_ADDITIONAL_STORES=/global/cfs/cdirs/m4385/podman_images
# export PODMANHPC_ADDITIONAL_STORES=/pscratch/sd/m/masao/roman_snpit/podman_images


# 2026-02-19 MWV:
# Testing 0.1.36 for Knop
ENV_VER=0.1.36

# ENV_VER=0.1.38

# ENV_VER=0.1.32  # 2026-02-12 current version
# Searching backward for a version with cuda 12, cupy 12 and gcc < 14
# Yes, that's 0.1.30
#ENV_VER=0.1.30

# CUDA 13.6, cupy 13.6
# 0.1.32 : E   RuntimeError: CuPy failed to load libnvrtc.so.12: OSError: libnvrtc.so.12: cannot open shared object file: No such file or directory
#
# cupy_backends/cuda/_softlink.pyx:32: RuntimeError
# 0.1.31 : Works  CUDA 12.4, cupy 13.5.1
# 0.1.30 : Works  CUDA 12.4, cupy 13.5.1
# 0.1.29 :
# 0.1.28 :
# 0.1.27 :

if [ "$WHICHROMANENV" = "cuda" ] || [ "$WHICHROMANENV" == "cuda-dev" ]; then
    PODMAN_HPC_RUN_GPU="--gpu"
else
    PODMAN_HPC_RUN_GPU=""
fi

podman-hpc run ${PODMAN_HPC_RUN_GPU} \
    --mount type=bind,source=$PWD,target=/home \
    --mount type=bind,source=$HOME/secrets,target=/secrets \
    --mount type=bind,source=$PSCRATCH/snpit_temp,target=/snpit_temp \
    --mount type=bind,source=$PSCRATCH,target=/scratch \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/m4385/env,target=/snpit_env \
    --mount type=bind,source=/pscratch/sd/m/masao/roman_snpit,target=/roman_snpit_masao_scratch \
    --mount type=bind,source=/pscratch/sd/m/masao/roman_snpit/database_dirs_rknop_dev,target=/data \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data,target=/ou2024 \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/PQ+HDF5_ROMAN+LSST_LARGE,target=/ou2024_snana \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/ROMAN+LSST_LARGE_SNIa-normal,target=/ou2024_snana_lc_dir \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/lsst/www/DESC_TD_PUBLIC/Roman+DESC/sims_sed_library,target=/ou2024_sims_sed_library \
    --mount type=bind,source=/dvs_ro/cfs/cdirs/m4385/calib_data/A25ePSF,target=/a25epsf \
    --mount type=bind,source=/global/cfs/cdirs/m4385/romanisim_sims,target=/romanimsim_sims \
    --env LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs \
    --env PYTHONPATH=/roman_imsim \
    --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env NUMEXPR_NUM_THREADS=1 \
    --env OMP_NUM_THREADS=1 \
    --env VECLIB_MAXIMUM_THREADS=1 \
    --env TERM=xterm \
    --env SNPIT_DEFAULT_CONFIG=/snpit_env/configs/rknop_dev_container_config.yaml \
    --env SNPIT_CONFIG=/snpit_env/configs/rknop_dev_container_config.yaml \
    --annotation run.oci.keep_original_groups=1 \
    -it \
    registry.nersc.gov/m4385/roman-snpit-env:${WHICHROMANENV:-cpu}-${ENV_VER} \
    /bin/bash $1
