# "sidecar" overview

The supernova detection pipeline for the Roman Supernova Project Infrastructure Team (SNPIT).  This package is designed to performance image difference, detect transient sources, and evaluate efficiency and purity for supernova science with the Roman Wide Field Instrument.

This package is intended to be run within the roman-snpit-env container environment. The instructions below walk you through preparing your environment and running the pipeline step-by-step on Perlmutter at NERSC. 

The name "sidecar" (which we could pretend is an acronym for "Supernova Ia DEteCtion AlgoRithm") is following the beverage-theme of Roman Supernova PIT photometry package naming.  It also a broader meaning of a supporting package to the main photometry.  We need to be able to find supernova in order to do photometry.  However, if we get identified targets from a different sources, the sidecar isn't necessary to then do the photometry.

## Step 1: Setup Container Environment

This package is meant to be run within the Roman Supernova PIT environment.  Follow the appropriate directions on the following page to set up your environment on the system you are running on:

https://roman-supernova-pit.github.io/snappl/environment.html

## Step 2: Clone `sidecar`

Run the following in terminal:

```bash
git clone https://github.com/Roman-Supernova-PIT/sidecar.git
```

Once cloning is complete, navigate to the 'sidecar' directory.

## Step 3: Detection

To run the detection pipeline, run the following code in `sidecar` repo of the terminal.

```
python sidecar/pipeline.py --image-collection [image collection] --data-records [path of the input file] --output-dir [output directory]
```

E.g.,

```
python sidecar/pipeline.py --image-collection ou2024 --data-records tests/test_one_data_record.csv --output-dir /dia_out_dir
```

Image collection is a string specifying an image collection that snappl knows about.
E.g., "ou2024", "manual_fits".  This will be used to generate paths for image info (observation_id, sca, band), get the correct objects that know how to load WCS and PSF information.

Note the `/dia_out_dir` only makes sense because we're running in the podman container, where we have bound /dia_out_dir to an output directory.

Can run by specifying the values on the command line
```
python sidecar/pipeline.py --image-collection ou2024 --observation-id 53526 --sca 1 --band R062 --template-observation-id 5044 --template-sca 8 --template-band R062 --output-dir /dia_out_dir
```

Can also run by just specifying the observation_id, sca, band of the science image

```
python sidecar/pipeline.py --image-collection ou2024 --observation-id 53526 --sca 1 --band R062 --output-dir /dia_out_dir
```

or just by passing the image path

```
image_path=/dvs_ro/cfs/cdirs/lsst/shared/external/roman-desc-sims/Roman_data/RomanTDS/images/simple_model/R062/53526/Roman_TDS_simple_model_R062_53526_1.fits.gz
python sidecar/pipeline.py --image-collection ou2024 --science-path ${image_path} --output-dir /dia_out_dir
```

Nov 2025 example
```
python sidecar/pipeline.py --image-collection snpitdb --image-provenance-tag ou2024 --image-process load_ou2024_image --observation-id 36846 --sca 15 --band H158 --output-dir /dia_out_dir
```

```
python sidecar/pipeline.py --image-collection snpitdb --image-provenance-tag ou2024 --image-process load_ou2024_image --science-observation-id 35303 --science-sca 8 --science-band H158 --template-observation-id 39140 --template-sca 3 --template-band H158 --output-dir /dia_out_dir
```

## ASDF The 49 example

After git cloning `sidecar` as above, see `sidecar/examples/perlmutter/asdf49_wfi01_run_one.sh` for an example of running

We need to run on a compute note to make sure we can use all of the GPU memory:

```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account m4385
```

Once you get your allocation and have a terminal on the node.

```
WHICHROMANENV=cuda-dev bash /global/cfs/cdirs/m4385/env/interactive-podman-rknop-dev.sh
cd /home/sidecar; pip install -e . --no-deps
cd /home
```

Then run the subtraction with:

```
obsid=99999010010010010010002
band=F106
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
    --no-reject-known-stars \
    --output-dir /snpit_temp/dia_out_dir/test_snappl \
    --temp-dir /snpit_temp
```

Running simple tests w/o database

For iterative CI testing or just ensuring basic functionality, there are two test scripts that are runnable on NERSC that subtract (1) two FITS files from OU224, and (2) two ASDF files "The 49" set of simulations.

```
sidecar/examples/perlmutter/test_sidecar_fits_manualou2024.sh
sidecar/examples/perlmutter/test_sidecar_asdf_manualrdm.sh
```


## sidecar Workflow
<img src="workflow.png" alt="Workflow of the detection pipeline." style="width:800px; height:auto;">

- Input: The pipeline takes a `csv` file with 6 required columns. They will be used as data ids to identify science and template images. During running, the pipeline will loop over each row to perform image difference, source detection, truth retrieval, and truth matching.
  ```
  | science_band | science_observation_id | science_sca | template_band | template_observation_id | template_sca |
  ```
- Subtraction: Perform image difference using [SFFT](https://github.com/thomasvrussell/sfft) algorithm.
- Detection: Perform source detection using [Source-Extractor](https://sextractor.readthedocs.io/en/latest/Introduction.html).
- Truth Retrieval: Retrieve truth tables of the science image and template image.
- Truth Matching: Match the truth information to the detected sources for evaluating efficiency. Match the detected sources to truth for evaluating purity.
- Evaluation: Evaluate efficiency and purity.
