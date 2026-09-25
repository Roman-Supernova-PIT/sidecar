#!/usr/bin/env python3
"""User: BCN.

Date: Jan 30 2026
Merged script: Makes cutouts on-the-fly and adds CNN predictions to ECSV catalog.
No cutout files are saved - only predictions are added to the catalog.

Expected directory structure:
    dia_out_dir/
        R062_56618_18_-_R062_2344_8/
            cleaned_score_detection_R062_56618_18_-_R062_2344_8.ecsv
            decorr_diff_R062_56618_18_-_R062_2344_8.fits
        (other similar subdirectories...)

ECSV format expected:
    Columns: id, x_peak, y_peak, peak_value, x_centroid, y_centroid, ra, dec

Output:
    Saves: cleaned_score_detection_<folder_name>_with_predictions.ecsv
    Added columns: cnn_prediction (1/0/-1), cnn_probability (0-1 or NaN)
    Added metadata: cnn_threshold, cnn_model, cnn_cutout_size
"""

import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import ascii, fits
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CNN MODEL DEFINITION (from CNN_prediction.py)
# ============================================================================

class _DenseLayer(nn.Module):
    """Dense layer implementation following original DenseNet paper."""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def bn_function(self, inputs):
        """Bottleneck function."""
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        """Forward pass."""
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    """Dense block implementation following original DenseNet paper."""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, init_features):
        """Forward pass."""
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """Transition layer implementation following original DenseNet paper."""

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """DenseNet-BC model implementation."""

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features,
                                                     kernel_size=7, stride=2,
                                                     padding=3, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()


def densenet169(**kwargs):
    """DenseNet-169 model."""
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)


# ============================================================================
# CUTOUT GENERATION FUNCTIONS (from make_cutouts.py)
# ============================================================================

def normalize_cutout_0_1(cutout_data):
    """Normalize cutout to 0-1 range."""
    valid_mask = np.isfinite(cutout_data)
    if not np.any(valid_mask):
        return np.zeros_like(cutout_data)

    data_min = np.min(cutout_data[valid_mask])
    data_max = np.max(cutout_data[valid_mask])

    if data_max == data_min:
        return np.zeros_like(cutout_data)

    normalized = (cutout_data - data_min) / (data_max - data_min)
    normalized[~valid_mask] = 0
    return normalized


def extract_cutout_with_padding(image_data, center_x, center_y, cutout_size=64, pad_value=0):
    """Extract cutout with padding if it goes beyond image boundaries."""
    half_size = cutout_size // 2

    # Calculate desired bounds
    y_start = center_y - half_size
    y_end = center_y + half_size
    x_start = center_x - half_size
    x_end = center_x + half_size

    # Calculate actual bounds (clipped to image)
    y_start_img = max(0, y_start)
    y_end_img = min(image_data.shape[0], y_end)
    x_start_img = max(0, x_start)
    x_end_img = min(image_data.shape[1], x_end)

    # Create padded cutout
    cutout = np.full((cutout_size, cutout_size), pad_value, dtype=image_data.dtype)

    # Calculate where to place the extracted data in the cutout
    y_start_cut = y_start_img - y_start
    y_end_cut = y_start_cut + (y_end_img - y_start_img)
    x_start_cut = x_start_img - x_start
    x_end_cut = x_start_cut + (x_end_img - x_start_img)

    # Extract and place data
    cutout[y_start_cut:y_end_cut, x_start_cut:x_end_cut] = \
        image_data[y_start_img:y_end_img, x_start_img:x_end_img]

    return cutout


def detect_catalog_type(data):
    """Detect which type of catalog this is based on column names.

    For cleaned_score_detection files with columns like:
    id, x_peak, y_peak, peak_value, x_centroid, y_centroid, ra, dec
    """
    colnames = data.colnames

    # Check for score catalog (x_centroid, y_centroid) - this is our target format
    if 'x_centroid' in colnames and 'y_centroid' in colnames:
        id_col = 'id' if 'id' in colnames else 'NUMBER'
        return 'score', 'x_centroid', 'y_centroid', id_col

    # Check for detection catalog (X_IMAGE, Y_IMAGE) - legacy support
    elif 'X_IMAGE' in colnames and 'Y_IMAGE' in colnames:
        id_col = 'NUMBER' if 'NUMBER' in colnames else 'id'
        return 'detection', 'X_IMAGE', 'Y_IMAGE', id_col

    else:
        raise ValueError(f"Unknown catalog type. Columns: {colnames}")


def determine_fits_basename(folder_name):
    """Determine the corresponding FITS filename from folder name.

    Example: folder_name = 'R062_56618_18_-_R062_2344_8'
             returns 'decorr_diff_R062_56618_18_-_R062_2344_8'
    """
    return f'decorr_diff_{folder_name}'


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_cutout(model, cutout_normalized, device, threshold=0.5):
    """Make prediction on a single cutout."""
    # Convert to tensor and replicate to 3 channels
    image_tensor = torch.from_numpy(cutout_normalized).float()
    image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)

    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probability = output.cpu().numpy().item()
        prediction = 1 if probability >= threshold else 0
        return prediction, probability


def process_single_catalog(catalog_path, fits_path, output_path,
                            model_path='DenseNet169_best.pth',
                            cutout_size=64, threshold=0.5):
    """Process a single ECSV catalog file with CNN predictions.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the input ECSV catalog file
    fits_path : str or Path
        Path to the corresponding FITS difference image
    output_path : str or Path
        Path where the output ECSV with predictions should be saved
    model_path : str
        Path to the trained model file
    cutout_size : int
        Size of cutouts to extract
    threshold : float
        Decision threshold for classification
    """
    device = torch.device('cpu')

    # Load model
    model = densenet169(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load catalog
    data = ascii.read(str(catalog_path), format='ecsv')
    catalog_type, x_col, y_col, id_col = detect_catalog_type(data)

    # Load FITS image
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data
        if image_data.ndim > 2:
            image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()

    # Process each object
    predictions = []
    probabilities = []
    half_size = cutout_size // 2

    for idx in range(len(data)):
        row = data[idx]
        try:
            x_image = float(row[x_col])
            y_image = float(row[y_col])

            if x_col in ['X_IMAGE', 'Y_IMAGE']:
                x_center = int(x_image - 1)
                y_center = int(y_image - 1)
            else:
                x_center = int(x_image)
                y_center = int(y_image)

            if (x_center < -half_size or x_center >= image_data.shape[1] + half_size or
                    y_center < -half_size or y_center >= image_data.shape[0] + half_size):
                predictions.append(-1)
                probabilities.append(np.nan)
                continue

            cutout_raw = extract_cutout_with_padding(
                image_data, x_center, y_center, cutout_size, pad_value=0
            )

            if np.all(cutout_raw == 0):
                predictions.append(-1)
                probabilities.append(np.nan)
                continue

            cutout_normalized = normalize_cutout_0_1(cutout_raw)
            prediction, probability = predict_cutout(model, cutout_normalized,
                                                     device, threshold)
            predictions.append(prediction)
            probabilities.append(probability)

        except Exception:
            predictions.append(-1)
            probabilities.append(np.nan)

    # Add predictions to catalog
    data['cnn_prediction'] = predictions
    data['cnn_probability'] = probabilities
    data.meta['cnn_threshold'] = threshold
    data.meta['cnn_model'] = model_path
    data.meta['cnn_cutout_size'] = cutout_size

    # Save
    ascii.write(data, output_path, format='ecsv', overwrite=True)


def process_ecsv_with_predictions(dia_out_dir='../dia_out_dir',
                                   model_path='DenseNet169_best.pth',
                                   cutout_size=64,
                                   threshold=0.5,
                                   allow_edge_cutouts=True):
    """Process ECSV files, make predictions on-the-fly, and save updated catalogs."""

    # Device configuration - force CPU only
    device = torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Decision threshold: {threshold}")

    # Load model
    print(f"Loading model from {model_path}")
    model = densenet169(num_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully\n")

    dia_out_path = Path(dia_out_dir)

    if not dia_out_path.exists():
        print(f"Directory not found: {dia_out_dir}")
        return

    # Find all subdirectories
    subdirs = [d for d in dia_out_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {dia_out_dir}")
        return

    print("=" * 70)
    print(f"Found {len(subdirs)} subdirectories")
    print("=" * 70)

    total_processed = 0
    total_skipped = 0

    for subdir_idx, subdir in enumerate(subdirs, 1):
        print(f"\n{'='*70}")
        print(f"[{subdir_idx}/{len(subdirs)}] {subdir.name}")
        print(f"{'='*70}")

        # Look for cleaned_score_detection_<folder_name>.ecsv in the subdirectory
        folder_name = subdir.name
        ecsv_file = subdir / f'cleaned_score_detection_{folder_name}.ecsv'

        if not ecsv_file.exists():
            print(f"ECSV file not found: {ecsv_file.name}")
            continue

        print(f"Found ECSV file: {ecsv_file.name}")

        try:
            data = ascii.read(str(ecsv_file), format='ecsv')
            print(f"Loaded ECSV with {len(data):,} objects")

            # Detect catalog type
            catalog_type, x_col, y_col, id_col = detect_catalog_type(data)
            print(f"Detected catalog type: '{catalog_type}'")
            print(f"Using columns: X={x_col}, Y={y_col}, ID={id_col}")

            # Show sample of data
            print("  Sample data:")
            for i in range(min(3, len(data))):
                obj_id = data[i][id_col] if id_col in data.colnames else i
                print(f"    Object {obj_id}: {x_col}={data[i][x_col]:.2f}, {y_col}={data[i][y_col]:.2f}")

        except Exception:
            print("Error reading ECSV")
            continue

        # Determine FITS filename from folder name
        fits_basename = determine_fits_basename(folder_name)
        fits_file = subdir / f"{fits_basename}.fits"

        if not fits_file.exists():
            print(f"FITS file not found: {fits_file.name}")
            continue

        # Load FITS image
        try:
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                if image_data.ndim > 2:
                    image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
                print(f"Loaded FITS: {fits_file.name} (shape: {image_data.shape})")
        except Exception:
            print("Error reading FITS")
            continue

        # Initialize prediction columns
        predictions = []
        probabilities = []
        successful = 0
        skipped = 0

        half_size = cutout_size // 2

        print(f"Processing {len(data):,} objects...")
        for idx in tqdm(range(len(data)), desc="Making predictions", ncols=100):
            row = data[idx]

            try:
                # Extract coordinates
                x_image = float(row[x_col])
                y_image = float(row[y_col])

                # Convert to 0-indexed
                if x_col in ['X_IMAGE', 'Y_IMAGE']:
                    x_center = int(x_image - 1)
                    y_center = int(y_image - 1)
                else:
                    # Centroids are typically already 0-indexed
                    x_center = int(x_image)
                    y_center = int(y_image)

                # Check if completely outside image
                if (x_center < -half_size or x_center >= image_data.shape[1] + half_size or
                        y_center < -half_size or y_center >= image_data.shape[0] + half_size):
                    predictions.append(-1)
                    probabilities.append(np.nan)
                    skipped += 1
                    continue

                # Extract cutout with padding
                cutout_raw = extract_cutout_with_padding(
                    image_data, x_center, y_center, cutout_size, pad_value=0
                )

                # Skip if cutout is all padding
                if np.all(cutout_raw == 0):
                    predictions.append(-1)
                    probabilities.append(np.nan)
                    skipped += 1
                    continue

                # Normalize cutout
                cutout_normalized = normalize_cutout_0_1(cutout_raw)

                # Make prediction
                prediction, probability = predict_cutout(model, cutout_normalized,
                                                         device, threshold)

                predictions.append(prediction)
                probabilities.append(probability)
                successful += 1

            except Exception:
                predictions.append(-1)
                probabilities.append(np.nan)
                skipped += 1
                continue

        # Add prediction columns to catalog
        data['real'] = predictions
        data['real_probability'] = probabilities

        # Add metadata about the threshold used
        data.meta['cnn_threshold'] = threshold
        data.meta['cnn_model'] = model_path
        data.meta['cnn_cutout_size'] = cutout_size

        # Save updated catalog
        output_file = subdir / f"cleaned_score_detection_{folder_name}_with_predictions.ecsv"
        ascii.write(data, output_file, format='ecsv', overwrite=True)

        print(f"\n{'─'*70}")
        print(f"Summary for {folder_name}:")
        print(f"Successful predictions: {successful:,}")
        print(f"Skipped (edge/invalid): {skipped:,}")
        print(f"Positive detections: {sum(p == 1 for p in predictions):,}")
        print(f"Negative detections: {sum(p == 0 for p in predictions):,}")
        print(f"Saved to: {output_file.name}")
        print(f"{'─'*70}")

        total_processed += successful
        total_skipped += skipped

        del image_data

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Total predictions: {total_processed:,}")
    print(f"Total skipped: {total_skipped:,}")
    print(f"{'='*70}")


# ============================================================================
# ENSEMBLE (Roman-Supernova-PIT/transient-real-bogus on Hugging Face)
# 6 architecture families x 4 members = 24 models.
#
# IMPORTANT: this ensemble was trained with sub-pixel bilinear cutouts and a
# DOUBLE z-scale normalization (full image, then cutout again) -- see
# normalize_with_zscale/create_cutout below. Do NOT swap these for
# normalize_cutout_0_1/extract_cutout_with_padding above; those use plain
# min-max on an integer-pixel cutout, which is a different convention the
# ensemble was not trained on and will silently degrade predictions.
# ============================================================================

REPO_ID = "Roman-Supernova-PIT/transient-real-bogus"
ALL_FAMILIES = ["DenseNet169", "ResNeXt50", "RegNetY016",
                 "EfficientNetB0", "ConvNeXtTiny", "DeiTTiny"]
TIMM_NAMES = {
    "ResNeXt50": "resnext50_32x4d",
    "RegNetY016": "regnety_016",
    "EfficientNetB0": "efficientnet_b0",
    "ConvNeXtTiny": "convnext_tiny",
}


def normalize_with_zscale(data):
    """ZScale then min-max to [0, 1]. Matches training_script.py exactly."""
    from astropy.visualization import ZScaleInterval

    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    zscale = ZScaleInterval()
    try:
        vmin, vmax = zscale.get_limits(data[valid_mask])
        if vmax > vmin:
            normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(data)
        normalized[~valid_mask] = 0
    except Exception:
        if data.max() > data.min():
            normalized = (data - data.min()) / (data.max() - data.min())
        else:
            normalized = np.zeros_like(data)
    return normalized.astype(np.float32)


def create_cutout(data, center_y, center_x, cutout_size=64):
    """Square cutout centered at (center_y, center_x), sub-pixel-aware
    (bilinear interpolation), zero-padded at edges."""
    half_size = cutout_size // 2
    y_start = center_y - half_size
    x_start = center_x - half_size
    y_coords = np.arange(cutout_size) + y_start
    x_coords = np.arange(cutout_size) + x_start

    cutout = np.zeros((cutout_size, cutout_size), dtype=data.dtype)
    max_y, max_x = data.shape[0] - 1, data.shape[1] - 1

    for i in range(cutout_size):
        for j in range(cutout_size):
            y_pos, x_pos = y_coords[i], x_coords[j]
            if 0 <= y_pos <= max_y - 1 and 0 <= x_pos <= max_x - 1:
                y_floor, x_floor = int(np.floor(y_pos)), int(np.floor(x_pos))
                y_ceil, x_ceil = min(max_y, y_floor + 1), min(max_x, x_floor + 1)
                y_floor, x_floor = max(0, min(max_y, y_floor)), max(0, min(max_x, x_floor))
                dy, dx = y_pos - y_floor, x_pos - x_floor
                try:
                    cutout[i, j] = (data[y_floor, x_floor] * (1 - dy) * (1 - dx) +
                                     data[y_ceil, x_floor] * dy * (1 - dx) +
                                     data[y_floor, x_ceil] * (1 - dy) * dx +
                                     data[y_ceil, x_ceil] * dy * dx)
                except IndexError:
                    safe_y = max(0, min(max_y, int(round(y_pos))))
                    safe_x = max(0, min(max_x, int(round(x_pos))))
                    cutout[i, j] = data[safe_y, safe_x]
            elif 0 <= y_pos <= max_y and 0 <= x_pos <= max_x:
                safe_y = max(0, min(max_y, int(round(y_pos))))
                safe_x = max(0, min(max_x, int(round(x_pos))))
                cutout[i, j] = data[safe_y, safe_x]

    return cutout


def load_fits_2d_ensemble(fits_path):
    """Read a 2D image from a FITS file, preferring the DATA/SCI extension
    (decorr_diff_*.fits keeps the image there, not in PRIMARY -- hdul[0].data
    is None for these files, unlike the plain hdul[0].data reads above)."""
    with fits.open(fits_path) as hdul:
        data = None
        for ext_name in ("DATA", "SCI"):
            if ext_name in hdul:
                data = hdul[ext_name].data
                break
        if data is None:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break
        if data is None:
            raise ValueError(f"No image data found in {fits_path}")
        data = np.asarray(data, dtype=np.float64).copy()
    if data.ndim > 2:
        data = data[0] if data.ndim == 3 else data.squeeze()
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def to_model_input(cutout_2d):
    return torch.from_numpy(cutout_2d).float().unsqueeze(0).repeat(3, 1, 1)


class TimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes=1, dropout=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=False,
                                           num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.backbone(x))).squeeze(1)


class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        import timm
        self.deit = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                                       num_classes=0, global_pool="avg", img_size=64)
        nf = self.deit.num_features
        self.classifier = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return torch.sigmoid(self.classifier(self.deit(x))).squeeze(1)


def build_ensemble_model(family):
    if family == "DenseNet169":
        return densenet169(num_classes=1)  # reuses this file's existing DenseNet
    if family == "DeiTTiny":
        return DeiTClassifier()
    return TimmClassifier(TIMM_NAMES[family])


def family_is_cached(model_dir, family):
    return len(glob.glob(os.path.join(model_dir, family, f"{family}_Ensemble_Model*_best.pth"))) == 4


def ensure_models_downloaded(model_dir, families):
    missing = [fam for fam in families if not family_is_cached(model_dir, fam)]
    if not missing:
        return model_dir
    from huggingface_hub import snapshot_download
    patterns = ["README.md"] + [p for fam in missing for p in (f"{fam}/*.pth", f"{fam}/README.md")]
    return snapshot_download(repo_id=REPO_ID, local_dir=model_dir, allow_patterns=patterns)


def load_ensemble(model_dir, families, device):
    """Download (if needed) + load every requested family.
    Returns dict {family_name: [models]}."""
    model_dir = ensure_models_downloaded(model_dir, families)
    family_models = {}
    for fam in families:
        models = []
        for path in sorted(glob.glob(os.path.join(model_dir, fam, f"{fam}_Ensemble_Model*_best.pth"))):
            model = build_ensemble_model(fam).to(device)
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
        if models:
            family_models[fam] = models
    if not family_models:
        raise RuntimeError("No models loaded -- check model_dir / families")
    return family_models


def predict_ensemble(family_models, cutout_2d_normalized, device):
    """"Ensemble of ensembles": each family's members are averaged into one
    per-family score, then the final score is the mean of those per-family
    scores. Returns (per_family_mean: dict, overall_mean: float, overall_std: float).
    overall_std is the spread across ALL individual model outputs."""
    x = to_model_input(cutout_2d_normalized).unsqueeze(0).to(device)
    per_family_mean = {}
    all_raw = []
    with torch.no_grad():
        for fam, models in family_models.items():
            probs = [m(x).item() for m in models]
            per_family_mean[fam] = float(np.mean(probs))
            all_raw.extend(probs)
    overall_mean = float(np.mean(list(per_family_mean.values())))
    overall_std = float(np.std(all_raw))
    return per_family_mean, overall_mean, overall_std


def process_single_catalog_ensemble(catalog_path, fits_path, output_path,
                                     model_dir=None, families=None,
                                     cutout_size=64, threshold=0.5):
    """Ensemble equivalent of process_single_catalog: makes cutouts on-the-fly
    (no cutout files saved) and adds real/bogus ensemble predictions to the
    ECSV catalog, using the 24-model Roman-Supernova-PIT/transient-real-bogus
    ensemble instead of the single local DenseNet169.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the input ECSV catalog file
    fits_path : str or Path
        Path to the corresponding FITS difference image
    output_path : str or Path
        Path where the output ECSV with predictions should be saved
    model_dir : str or Path
        Where the ensemble weights live (downloaded from Hugging Face on
        first use, cached and skipped on later calls). Defaults to
        ~/roman_sn_pit_models.
    families : list of str
        Subset of ALL_FAMILIES to use. Defaults to all 6.
    cutout_size : int
        Size of cutouts to extract
    threshold : float
        Decision threshold on the overall real_bogus_score
    """
    if model_dir is None:
        model_dir = os.path.expanduser("~/roman_sn_pit_models")
    if families is None:
        families = ALL_FAMILIES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_models = load_ensemble(model_dir, families, device)
    families_order = [f for f in families if f in family_models]
    total_models = sum(len(v) for v in family_models.values())

    data = ascii.read(str(catalog_path), format='ecsv')
    catalog_type, x_col, y_col, id_col = detect_catalog_type(data)

    normalized_full = normalize_with_zscale(load_fits_2d_ensemble(fits_path))
    half_size = cutout_size // 2

    per_family_cols = {fam: [] for fam in families_order}
    scores, stds, preds = [], [], []

    for idx in tqdm(range(len(data)), desc="Scoring (ensemble)"):
        row = data[idx]
        try:
            x_image = float(row[x_col])
            y_image = float(row[y_col])
            if x_col in ('X_IMAGE', 'Y_IMAGE'):
                x_center, y_center = x_image - 1.0, y_image - 1.0
            else:
                x_center, y_center = x_image, y_image

            if (x_center < -half_size or x_center >= normalized_full.shape[1] + half_size or
                    y_center < -half_size or y_center >= normalized_full.shape[0] + half_size):
                raise ValueError("center outside image bounds")

            cutout = create_cutout(normalized_full, y_center, x_center, cutout_size)
            cutout = normalize_with_zscale(cutout.astype(np.float32))
            per_family_mean, overall_mean, overall_std = predict_ensemble(family_models, cutout, device)

            for fam in families_order:
                per_family_cols[fam].append(per_family_mean[fam])
            scores.append(overall_mean)
            stds.append(overall_std)
            preds.append("real" if overall_mean >= threshold else "bogus")

        except Exception:
            for fam in families_order:
                per_family_cols[fam].append(np.nan)
            scores.append(np.nan)
            stds.append(np.nan)
            preds.append("invalid")

    for fam in families_order:
        data[f"rb_{fam}"] = per_family_cols[fam]
    data["real_bogus_score"] = scores
    data["real_bogus_score_std"] = stds
    data["prediction"] = preds
    data.meta["real_bogus_threshold"] = threshold
    data.meta["real_bogus_model_dir"] = str(model_dir)
    data.meta["real_bogus_n_models"] = total_models
    data.meta["real_bogus_cutout_size"] = cutout_size

    ascii.write(data, output_path, format='ecsv', overwrite=True)


if __name__ == "__main__":
    process_ecsv_with_predictions(
        dia_out_dir='../dia_out_dir',
        model_path='DenseNet169_best.pth',
        cutout_size=64,
        threshold=0.5,
        allow_edge_cutouts=True
    )
    print("\nComplete!")
