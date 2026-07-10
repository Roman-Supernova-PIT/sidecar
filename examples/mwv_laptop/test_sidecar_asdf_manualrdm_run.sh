SNPIT_PIPELINE_BASE=${HOME}/Roman/pipeline

# cd ${SNPIT_PIPELINE_BASE}/sidecar; pip install -e . --no-deps
# cd ${SNPIT_PIPELINE_BASE}

base_path=${SNPIT_PIPELINE_BASE}/photometry_test_data/asdf_the_49
template_path=r9999901001001001001_0001_wfi01_f158_cal.asdf
science_path=r9999901001001001001_0002_wfi01_f158_cal.asdf

python \
    sidecar/sidecar/pipeline.py \
    --image-collection manual_rdm \
    --base-path ${base_path} \
    --template-path ${template_path} \
    --science-path ${science_path} \
    --no-reject-known-stars \
    --no-cross-convolve \
    --temp-dir ${HOME}/tmp \
    --output-dir ./ \
    --backend4subtract numpy
