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

### Installing outside the container (pip)

The default install is CPU-only. PyPI's Linux `torch` wheel bundles CUDA
(several GB of `nvidia-*` packages), so install the CPU builds of PyTorch
and torchvision (needed by `timm`) first; pip then sees they are already
satisfied and will not fetch the CUDA ones:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

For GPU (`--backend4subtract Cupy`), install the `gpu` extra instead, which
adds CuPy and the CUDA build of PyTorch:

```bash
pip install -e ".[gpu]"
```

## Real/bogus models (run once per system)

By default the pipeline scores detections with a CNN real/bogus ensemble
(`--real-bogus`). Pipeline jobs only *read* the models from
`photometry.sidecar.real_bogus.model_dir`, which defaults to
`${system.paths.dev_storage}/sidecar/real_bogus_models`. They never download
them, so many parallel jobs can share one copy without racing to write it.

### Getting access to the models

The models are in the private Hugging Face repository
[`Roman-Supernova-PIT/transient-real-bogus`](https://huggingface.co/Roman-Supernova-PIT/transient-real-bogus).
Because it is private, access is granted by adding you to the
`Roman-Supernova-PIT` Hugging Face organization:

1. Create a free Hugging Face account at https://huggingface.co/join.
2. Email your Hugging Face username to Dr. Benjamin Rose (ben_rose@baylor.edu) and
   ask to be added to the `Roman-Supernova-PIT` organization.
3. Once added, create a read-access token at
   https://huggingface.co/settings/tokens and log in with `hf auth login`
   (or `export HF_TOKEN=<your token>`).

### Downloading the models

Before launching a set of jobs on a system, download the models once into
that directory, from inside the SNPIT environment on a node with network
access:

```bash
# The Hugging Face repository is private: log in once or set HF_TOKEN.
hf auth login            # or: export HF_TOKEN=...
python -m sidecar.classifier --download-models
```

Families already present are skipped, so it is safe to run again. Use
`--model-dir DIR` to download somewhere else, and point
`photometry.sidecar.real_bogus.model_dir` at the same place. The directory
must be on storage that every job (container) mounts.

If the models are missing, the pipeline stops with an error explaining how to
download them. To run without real/bogus scoring, pass `--no-real-bogus`.

The CNNs run on the same device as the subtraction: `--backend4subtract Cupy`
uses the GPU, `numpy` the CPU.

### 3-channel (science, template, diff) models

A second ensemble,
[`Roman-Supernova-PIT/Real-Bogus-3ch`](https://huggingface.co/Roman-Supernova-PIT/Real-Bogus-3ch)
(DenseNet121 + ConvNeXtTiny, 4 members each, same access as above), scores
each detection on science, template and diff cutouts instead of the diff
alone. Download it once into `photometry.sidecar.real_bogus.model_dir_3ch`
(default `${system.paths.dev_storage}/sidecar/real_bogus_3ch_models`):

```bash
python -m sidecar.classifier --download-models --model-type 3ch
```

and select it with `photometry.sidecar.real_bogus.model: 3ch` in the config
(or `--photometry-sidecar-real_bogus-model 3ch` on the command line). The
subtraction then also writes `rb_science_*.fits` (sky-subtracted science) and
`rb_template_*.fits` (template resampled onto the science grid) to the output
directory, and the scores go to the same `real_bogus_*` catalog. Its
model version is `photometry.sidecar.real_bogus.version_3ch` (default
`nov_24_2026`, the folder in the Hugging Face repository; the models land in
`<model_dir_3ch>/<version_3ch>/`), and its
threshold is `photometry.sidecar.real_bogus.threshold_3ch` (default 0.5, not
yet tuned on sidecar subtractions).

Cut-and-paste example (SMDC):
[`examples/smdc/fetch_real_bogus_models.sh`](examples/smdc/fetch_real_bogus_models.sh)
downloads the models once, then
[`examples/smdc/asdf49_wfi01_run_one_real_bogus_slurm.sbatch`](examples/smdc/asdf49_wfi01_run_one_real_bogus_slurm.sbatch)
runs one subtraction with real/bogus scoring.

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
- Detection: Find positive and negative peaks in background-subtracted difference and score images using a
  local background-RMS threshold (5 times the local RMS by default); catalog `peak_value` is background-subtracted.
  Subtraction products store SFFT's blank-region mask in a separate `NO_COVERAGE` FITS extension.
- Cleaned catalogs are written when known-star rejection is enabled. By default, stars with Gaia G
  magnitude 16.5 or fainter use a 5 arcsec rejection radius; for brighter stars the radius is
  `min(60, 5 * 10**(0.2 * (16.5 - G)))` arcsec. Nonempty star catalogs must include
  `phot_g_mean_mag`; stars with missing magnitude values use the base radius.
  Configure these values with `--star-rejection-match-radius`, `--star-rejection-ref-mag`,
  `--star-rejection-scale`, and `--star-rejection-max-radius`. Set `--star-rejection-ref-mag none`
  for a fixed radius or `--star-rejection-max-radius none` for no cap. These options apply to all
  cleaned catalogs and are separate from the pixel-based bright-star masks used for SFFT source selection.
  The defaults were tested on one field and can also reject real transients close to stars; validate them
  for each dataset.
  Cleaned score and simple-difference
  catalogs also exclude sources with bad or unmeasured pixels in the 3-pixel flag aperture of the science,
  template, or respective subtraction product, each measured at the source's sky position.
  The cleaned score catalog additionally requires `abs(score_aper_r3_snr) >= 7`, configurable with
  `--cleaned-score-min-snr`; significant negative detections are retained. The simple-difference catalog
  has no S/N cut by default; `--cleaned-simple-min-snr` sets a minimum absolute r=3 aperture S/N.
  The score-image aperture S/N uses the difference-image noise plane and does not fully
  account for correlated noise, so this cut should be validated for each dataset. The product flag plane
  currently combines science and template flags without resampling the template flags to the science grid.
  Optionally, `--reject-center-jump-det` rejects `JUMP_DET` at the source center of any of these three
  images; `--center-dq-flags MASK` rejects any bit in an additional integer DQ mask (decimal or `0x` hex).
  Both are off by default. When enabled, unmeasured center flags are rejected as well.
  Override the checked columns with `--bad-pixel-columns` and `--center-flag-columns`, followed by
  space-separated names. Names beginning with `science_` or `template_` are used as-is; other names
  receive the `score_` or `simple_diff_` prefix, depending on the catalog. For example,
  `--bad-pixel-columns science_flags_npix_bad flags_npix_bad` checks science and the product.
  Supplying `--bad-pixel-columns` without names disables the aperture bad-pixel cut in both catalogs;
  `--center-flag-columns` requires at least one name.
  All three cleaned catalogs also apply circular and square single-pixel artifact cuts by default.
  These cuts run only when known-star rejection is enabled. For positive
  detections the cut uses the science measurements; for negative detections it uses the template.
  A candidate is rejected when its input-image r=3 S/N is at least 5, r=2/r=3 flux is at least 0.95,
  and peak/r=3 flux is at least 0.7. The peak is the maximum pixel within the r=2 aperture, after
  subtracting the local background. These limits can be changed with `--single-pixel-min-input-snr`,
  `--single-pixel-min-r2-r3`, and `--single-pixel-min-peak-r3`, respectively. The r=2, 3, and 5
  input apertures must have their full area to within `--single-pixel-area-atol` (default 0.1 square
  pixels); the detection peak and r=3 S/N must have the same sign. Candidates with incomplete,
  nonfinite, low-S/N, or sign-inconsistent measurements are retained and logged as inconclusive.
  Disable the cut with `--no-reject-single-pixel`. These thresholds were explored on one field and
  one SN candidate and should be validated against real sources, PSF variations, and injected sources.
  Detection catalogs also contain local-background-subtracted square fluxes for each measured image:
  `<image>_square_1_flux`, `_square_3_flux`, and `_square_5_flux` are whole-pixel sums in 1x1,
  3x3, and 5x5 squares centered on the pixel nearest the candidate's position in that image.
  Their `_flux_err` and `_snr` columns use the image noise plane and ignore correlations and
  background-estimation uncertainty. `_square_1_over_3` and `_square_3_over_5` are the corresponding
  flux ratios. A square with a bad, masked, nonfinite, or out-of-image pixel is unmeasured (NaN);
  ratios are NaN if either sum is unavailable or the denominator is nonpositive.
  The experimental square-flux cut flags rows with input-image
  3x3 S/N >= 20, 1x1/3x3 >= 0.8, and 3x3/5x5 >= 0.9, using science for positive and template for
  negative detections. Set the thresholds with `--square-pixel-min-snr`,
  `--square-pixel-min-1-over-3`, and `--square-pixel-min-3-over-5`. The product peak and r=3 S/N
  must agree in sign. Inconclusive measurements are retained and logged; disabling the square cut with
  `--no-reject-square-pixel` does not disable the circular cut, and vice versa. The square cut runs after
  the circular cut, and its experimental thresholds require broader validation.
- Standalone star rejection accepts ECSV or Parquet for both detections and reference stars:
  `python -m sidecar.reject_stars detections.parquet stars.ecsv`. Both this command and
  `sidecar.pipeline` accept `--catalog-output-format parquet|ecsv|both` (default `parquet`) for catalog
  outputs. `both` writes ECSV and a Parquet companion with the same stem; the ECSV path is
  the primary path. For standalone star rejection, `--output` must have the `.parquet` extension
  for `parquet`, or `.ecsv` for `ecsv` and `both`. Without `--output`, the name is
  `cleaned_<input stem>` with the appropriate extension. Pipeline truth, detection, star, cleaned,
  and matching catalogs use the selected format; FITS images are unaffected.
- Truth Retrieval: Retrieve truth tables of the science image and template image.
- Truth Matching: Match the truth information to the detected sources for evaluating efficiency. Match the detected sources to truth for evaluating purity.
- Evaluation: Evaluate efficiency and purity.

## Matching Catalogs Across Observations

To cross-match sources across detection catalogs on `ra` and `dec`, use `sidecar.match_catalogs`.
Entries must satisfy `template_in_bounds == True` by default. Additional boolean and numeric comparison cuts (`==`, `!=`, `>`, `<`, `>=`, `<=`) can be supplied with `--row-filter`.

### Example: Multi-Filter Matching

```bash
python -m sidecar.match_catalogs \
    ./subtractions/F*_15_-_F*_15/cleaned_score_*.ecsv \
    --min-matches 3 \
    --radius 1.0 \
    --row-filter "score_in_bounds,==,True" \
    --row-filter "score_peak_snr,>,10" \
    --row-filter "score_flags_npix_bad,<=,0" \
    --output matched_01062_cleaned_score_15.ecsv
```

In Python:

```python
from astropy.table import Table
import astropy.units as u
from sidecar.match_catalogs import match_catalogs

tables = [Table.read("cat1.ecsv"), Table.read("cat2.ecsv")]
matched = match_catalogs(
    tables,
    radius=1.0 * u.arcsec,
    min_matches=2,
    row_filters=[
        ("score_in_bounds", "==", True),
        ("score_peak_snr", ">", 10.0),
        ("score_flags_npix_bad", "<=", 0),
    ],
)
```
