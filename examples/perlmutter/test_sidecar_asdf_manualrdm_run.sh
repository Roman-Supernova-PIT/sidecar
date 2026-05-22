# cd /home/sidecar; pip install -e . --no-deps
# cd /home

base_path=/data/images/d83f716b-6425-f2bf-08df-e72ddeb3dc8b
template_path=r9999901001001001001_0001_wfi01_f158_cal.asdf
science_path=r9999901001001001001_0002_wfi01_f158_cal.asdf
python \
    sidecar/sidecar/pipeline.py \
    --image-collection manual_rdm \
    --base-path ${base_path} \
    --template-path ${template_path} \
    --science-path ${science_path} \
    --no-reject-known-stars \
    --output-dir ./
