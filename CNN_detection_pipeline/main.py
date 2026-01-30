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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import ascii, fits
from astropy.visualization import ZScaleInterval
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


def process_ecsv_with_predictions(dia_out_dir='../dia_out_dir',
                                   model_path='DenseNet169_best.pth',
                                   cutout_size=64,
                                   threshold=0.5,
                                   allow_edge_cutouts=True):
    """Process ECSV files, make predictions on-the-fly, and save updated catalogs."""
    
    # Device configuration
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

        except Exception as e:
            print(f"Error reading ECSV: {e}")
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
        except Exception as e:
            print(f"Error reading FITS: {e}")
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

            except Exception as e:
                predictions.append(-1)
                probabilities.append(np.nan)
                skipped += 1
                continue

        # Add prediction columns to catalog
        data['cnn_prediction'] = predictions
        data['cnn_probability'] = probabilities

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


if __name__ == "__main__":
    process_ecsv_with_predictions(
        dia_out_dir='../dia_out_dir',
        model_path='DenseNet169_best.pth',
        cutout_size=64,
        threshold=0.5,
        allow_edge_cutouts=True
    )
    print("\nComplete!")