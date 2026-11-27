# Run one subtraction + detection with CNN real/bogus scoring.
#
# Run fetch_real_bogus_models.sh once first, so the models are in
# photometry.sidecar.real_bogus.model_dir.  Without them this run stops with an error.
#
# The CNNs run on the same device as the subtraction: --backend4subtract numpy -> CPU,
# Cupy -> GPU.
#
# Output: real_bogus_score_detection_*.parquet in --output-dir, i.e. the score-image
# detections plus rb_<family>, real_bogus_score, real_bogus_score_std and prediction columns.

cd /home/snappl; pip install -e .
cd /home/sfft; pip install -e .
cd /home/sidecar; pip install -e . --no-deps
cd /home

# Can't reject known stars because we don't have a star catalog
obsid=99999010010010010010002
band=F106
python \
    /home/sidecar/sidecar/pipeline.py  \
    --image-collection snpitdb \
    --image-provenance-tag asdf_functional_test \
    --image-process load_rdm_image \
    --science-observation-id ${obsid} \
    --science-band ${band} \
    --science-sca 1 \
    --template-observation-id 99999010010010010010001 \
    --template-band ${band} \
    --template-sca 1 \
    --no-reject-known-stars \
    --real-bogus \
    --catalog-output-format parquet \
    --output-dir /sidecar_dia_out \
    --temp-dir /dev/shm \
    --backend4subtract numpy
