# CNN Detection Pipeline
## Directory Structure
```
/global/cfs/cdirs/m4385/users/wmwv/
├── pipeline/
│   └── CNN_detection_pipeline/           # Pipeline directory
│       ├── make_cutouts.py               # Step 1: Cutout extraction
│       ├── CNN_prediction.py             # Step 2: CNN predictions
│       ├── pipeline.py                   # Main pipeline script
│       ├── DenseNet121_best.pth          # Model weights (84 MB)
│       └── DenseNet169_best.pth          # Model weights (151 MB)
│
└── dia_out_dir/                          # DIA outputs (one level above)
    └── <diff_folder>/                    # e.g., decorr_diff_tile_name
        ├── cleaned_score_detection_to_transients_*.ecsv
        ├── decorr_diff_*.fits
        └── cutouts/                      # Created by pipeline
            └── fits/
                └── cutout_*.fits
        └── CNN_detection_results/        # Created by pipeline
            ├── positives/
            └── negatives/
```
## Quick Start
### 1. Navigate to Pipeline Directory
```bash
cd /global/cfs/cdirs/m4385/users/wmwv/pipeline/CNN_detection_pipeline/
```
### 2. Run Complete Pipeline
```bash
python3 pipeline.py
```
This single command runs both:
- **Step 1**: Cutout extraction from ECSV catalogs
- **Step 2**: CNN predictions on extracted cutouts
---
# Step 1: Run make_cutouts.py 
## General Usage:
```
python3 make_cutouts.py
```

# Step 2: Run CNN_prediction.py
## General Usage:
```
python3 CNN_prediction.py
```
This script processes the previously extracted cutouts through a DenseNet model:
1. Loads the trained DenseNet model (DenseNet169_best.pth by default).
2. Processes all FITS files in the cutouts directory using ZScale normalization.
3. Classifies each cutout as either a positive or negative detection.


# Git LFS
# Accessing Model Files with Git LFS
This repository uses **Git Large File Storage (Git LFS)** to store large model files `.pth` files) efficiently.
## Model Files
The following DenseNet model files are stored using Git LFS:
- `cnn_detection_pipeline/DenseNet121_best.pth` (84 MB)
- `cnn_detection_pipeline/DenseNet169_best.pth` (151 MB)
## Storage Location
The actual model files are stored on **GitHub's LFS servers**, not in the regular Git repository. When you clone or pull the repository, Git LFS automatically downloads these files from GitHub's LFS storage.
## How to Access the Files
### 1. Install Git LFS
First, install Git LFS on your system:
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
# macOS
brew install git-lfs
# Windows
# Download from: https://git-lfs.github.com/
```
### 2. Initialize Git LFS
```bash
git lfs install
```
### 3. Clone the Repository
When you clone the repository, Git LFS will automatically download the model files:
```bash
git clone https://github.com/Roman-Supernova-PIT/sidecar.git
cd cnn_detection_pipeline
```
The `.pth` files will be automatically downloaded to the correct locations.
### 4. Verify Files Were Downloaded
```bash
git lfs ls-files
```
You should see the model files listed. You can also check the file sizes:
```bash
ls -lh cnn_detection_pipeline/*.pth
```
## Authentication
**No special authentication is needed** beyond normal GitHub access:
If you already have access to clone this repository, you automatically have access to the LFS files.
## For Repository Maintainers
To track new large files with LFS:
```bash
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model files with LFS"
```
**Author**: [BC Nagam]
**Last Updated**: November 2025
**Python Version**: 3.8+
