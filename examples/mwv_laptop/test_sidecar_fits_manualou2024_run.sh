# cd /home/sidecar; pip install -e . --no-deps
# cd /home

base_path=${HOME}/Roman/pipeline/photometry_test_data/ou2024/images/simple_model
science_path="R062/54300/Roman_TDS_simple_model_R062_54300_13.fits.gz"
template_path="R062/1/Roman_TDS_simple_model_R062_1_2.fits.gz"
ipython --pdb -- \
    sidecar/sidecar/pipeline.py \
    --image-collection ou2024 \
    --base-path ${base_path} \
    --science-band R062 \
    --template-band R062 \
    --template-path ${template_path} \
    --science-path ${science_path} \
    --no-reject-known-stars \
    --temp-dir ${HOME}/tmp \
    --output-dir ./ \
    --backend4subtract numpy
