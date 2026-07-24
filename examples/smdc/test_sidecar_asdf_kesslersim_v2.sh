# cd ${HOME}/Roman/pipeline/snappl; pip install -e .
# cd ${HOME}/Roman/pipeline/sfft; pip install -e .
# cd ${HOME}/Roman/pipeline/sidecar; pip install -e . --no-deps
# cd ${HOME}/Roman/pipeline

export SNPIT_DEFAULT_CONFIG=${HOME}/Roman/pipeline/environment/smdc_interactive_config.yaml
export SNPIT_CONFIG=${SNPIT_DEFAULT_CONFIG}

base_path=/mnt/roman-science-east-2/snpit/snana+romanisim+romancal/output_images_SCAx2_ZYJHF_40day/

template_path=${base_path}/SNPIT_VISIT602000033_WFI01_F106_L2.asdf
science_path=${base_path}/SNPIT_VISIT607000033_WFI01_F106_L2.asdf
python -m pdb \
    sidecar/sidecar/pipeline.py \
    --image-collection manual_rdm \
    --base-path ${base_path} \
    --template-path ${template_path} \
    --science-path ${science_path} \
    --no-reject-known-stars \
    --temp-dir ${HOME}/tmp \
    --output-dir ./ \
    --save-debug-products \
    --backend4subtract numpy
