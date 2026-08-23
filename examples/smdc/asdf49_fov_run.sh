cd /home/sidecar; pip install -e . --no-deps
cd /home/snappl; pip install -e .
cd /home/sfft; pip install -e . --no-deps
cd /home

# Can't reject known stars because we don't have a star catalog
# This last combination of 99999010010010010010005 + F158 doesn't exist, so we only get 7 subtractions:
for obsid in 99999010010010010010030; do
  for band in F062; do
    for sca in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
      python \
        /home/sidecar/sidecar/pipeline.py  \
        --image-collection snpitdb \
        --image-provenance-tag asdf_functional_test \
        --image-process load_rdm_image \
        --science-observation-id ${obsid} \
        --science-band ${band} \
        --science-sca ${sca} \
        --template-observation-id 99999010010010010010020 \
        --template-band ${band} \
        --template-sca ${sca} \
        --no-reject-known-stars \
        --output-dir /sidecar_dia_out \
        --temp-dir /dev/shm \
        --backend4subtract numpy
    done
  done
done
