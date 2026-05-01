# cd /home/sidecar; pip install -e . --no-deps
# cd /home/snappl; pip install -e .
# cd /home/sfft; pip install -e . --no-deps

# Can't reject known stars because we don't have a star catalog
for obsid in 99999010010010010010002 99999010010010010010003 99999010010010010010004 99999010010010010010005; do
  for band in F106 F158; do
    python \
      sidecar/sidecar/pipeline.py  \
      --image-collection snpitdb \
      --image-provenance-tag asdf_functional_test \
      --image-process load_rdm_image \
      --science-observation-id ${obsid} \
      --science-band ${band} \
      --science-sca 1 \
      --template-observation-id 99999010010010010010001 \
      --template-band ${band} \
      --template-sca 1 \
      --reject-known-stars False \
      --output-dir /snpit_temp/dia_out_dir/test_snappl \
      --temp-dir /snpit_temp
  done
done
