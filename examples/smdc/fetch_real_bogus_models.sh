# Download the real/bogus CNN model ensemble ONCE per system, before launching jobs.
#
# Pipeline jobs only read the models from photometry.sidecar.real_bogus.model_dir
# (default ${system.paths.dev_storage}/sidecar/real_bogus_models) and never download
# them, so many jobs can share one copy.
#
# Run this on a node with network access, inside the container, e.g.
#   export HF_TOKEN=...   # the Hugging Face repository is private
#   cd ${HOME}/snpit/packages
#   bash environment/smdc-apptainer-mwv-dev.sh /home/sidecar/examples/smdc/fetch_real_bogus_models.sh
#
# Families already downloaded are skipped, so it is safe to run again.

cd /home/snappl; pip install -e .
cd /home/sfft; pip install -e . --no-deps
cd /home/sidecar; pip install -e . --no-deps
cd /home

python -m sidecar.classifier --download-models
# Optional: the 3-channel (science, template, diff) ensemble, used with
# photometry.sidecar.real_bogus.model: 3ch
python -m sidecar.classifier --download-models --model-type 3ch
