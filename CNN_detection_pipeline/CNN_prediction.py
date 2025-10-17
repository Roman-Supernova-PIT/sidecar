#!/usr/bin/env python3
"""User: BCN
Date: 14 Oct 2025

Prerequisite: make_cutouts.py should have been executed before this.
Such that, cutouts/<tile_name>/fits folder exists.
DenseNet169 FITS Inference Script with Correct ZScale Normalization

This script processes FITS files using a trained DenseNet169 model with
the same ZScale normalization used during training and cutout generation.

Requirements:
- torch
- torchvision
- matplotlib
- numpy
- astropy
- tqdm
- scikit-image
- pathlib

Install with: pip install torch torchvision matplotlib numpy astropy tqdm scikit-image
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from tqdm import tqdm
from skimage.transform import resize
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


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


def densenet121(**kwargs):
    """DenseNet-121 model."""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)


def normalize_with_zscale(data):
    """Normalize data using astropy's ZScale algorithm.

    This is the SAME function used in peak finder and injection cutouts scripts.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array

    Returns
    -------
    normalized_data : numpy.ndarray
        Normalized data between 0 and 1
    """
    # Handle invalid values
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)

    # Apply ZScale normalization
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data[valid_mask])

    # Normalize to 0-1 range
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)

    # Handle any remaining invalid values
    normalized[~valid_mask] = 0

    return normalized


def load_and_preprocess_fits(fits_path):
    """Load and preprocess a single FITS file with correct ZScale normalization."""
    try:
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)

            # Handle different data shapes
            if image_data.ndim > 2:
                image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()

            # Resize to 64x64 if needed
            if image_data.shape != (64, 64):
                image_data = resize(image_data, (64, 64), mode='constant', anti_aliasing=True)

            # Use ZScale normalization (SAME as training data)
            normalized_data = normalize_with_zscale(image_data)

            # Convert to tensor and replicate to 3 channels
            image_tensor = torch.from_numpy(normalized_data).float()
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)

            return image_tensor, normalized_data

    except Exception as e:
        print(f"Error loading file {fits_path}: {str(e)}")
        return None, None


def predict_single_image(model, image_tensor, device):
    """Make prediction on a single image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probability = output.cpu().numpy().item()
        prediction = 1 if probability >= 0.5 else 0
        return prediction, probability


def save_image_with_prediction(image_data, prediction, probability, original_path,
                                output_folder, metadata):
    """Save image as PNG with prediction info in filename and metadata overlay.

    Parameters
    ----------
    image_data : numpy.ndarray
        Image data to save
    prediction : int
        Prediction result (1=positive, 0=negative)
    probability : float
        Prediction probability
    original_path : str
        Path to original FITS file
    output_folder : str
        Folder to save output PNG file
    metadata : dict
        Dictionary containing extracted metadata (snr, magnitude, id, etc.)

    Returns
    -------
    str
        Path to saved PNG file
    """
    # Create output filename
    original_name = Path(original_path).stem

    # Determine prediction label
    pred_label = "POSITIVE" if prediction == 1 else "NEGATIVE"

    # Create filename with probability score
    output_filename = f"{original_name}_{pred_label}_p{probability:.4f}.png"
    output_path = os.path.join(output_folder, output_filename)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data, cmap='gray')

    # Create title with metadata
    title = f'Prediction: {pred_label}\nProbability: {probability:.4f}'
    if metadata['magnitude'] is not None:
        title += f'\nMagnitude: {metadata["magnitude"]:.3f}'
    if metadata['snr'] is not None:
        title += f'\nSNR: {metadata["snr"]:.2f}'
    if metadata['id'] is not None:
        title += f'\nID: {metadata["id"]}'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(label='Intensity (ZScale Normalized)')
    plt.axis('off')

    # Add colored border based on prediction
    border_color = 'green' if prediction == 1 else 'red'
    for spine in plt.gca().spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def process_fits_folder(input_folder, output_folder, model, device):
    """Process all FITS files in the input folder and save results.

    Parameters
    ----------
    input_folder : str
        Folder containing FITS files to process
    output_folder : str
        Folder to save results
    model : torch.nn.Module
        Trained DenseNet model
    device : torch.device
        Device to run model on (CPU or CUDA)

    Returns
    -------
    dict
        Dictionary containing processing results
    """
    # Create output folders
    positive_folder = os.path.join(output_folder, 'positives')
    negative_folder = os.path.join(output_folder, 'negatives')
    os.makedirs(positive_folder, exist_ok=True)
    os.makedirs(negative_folder, exist_ok=True)

    # Find all FITS files
    fits_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.fits', '.fit')):
                fits_files.append(os.path.join(root, file))

    print(f"Found {len(fits_files)} FITS files to process")

    # Process each file
    results = {
        'positive_count': 0,
        'negative_count': 0,
        'failed_count': 0,
        'results': []
    }

    for fits_path in tqdm(fits_files, desc="Processing FITS files"):
        # Extract metadata from filename
        filename = os.path.basename(fits_path)
        metadata = extract_metadata_from_filename(filename)

        # Load and preprocess with ZScale normalization
        image_tensor, image_data = load_and_preprocess_fits(fits_path)

        if image_tensor is None:
            results['failed_count'] += 1
            continue

        # Make prediction
        prediction, probability = predict_single_image(model, image_tensor, device)

        # Choose output folder based on prediction
        target_folder = positive_folder if prediction == 1 else negative_folder

        # Save image with prediction
        output_path = save_image_with_prediction(
            image_data, prediction, probability, fits_path, target_folder, metadata
        )

        # Update counts
        if prediction == 1:
            results['positive_count'] += 1
        else:
            results['negative_count'] += 1

        # Store result with metadata
        results['results'].append({
            'original_path': fits_path,
            'output_path': output_path,
            'prediction': prediction,
            'probability': probability,
            'snr': metadata['snr'],
            'magnitude': metadata['magnitude'],
            'id': metadata['id'],
            'x': metadata['x'],
            'y': metadata['y'],
            'band': metadata['band'],
            'subfolder': Path(fits_path).parent.name
        })

    return results


def extract_metadata_from_filename(filename):
    """Extract SNR and magnitude values from filename.

    Example: "psf_injection_diff_R062_41205_12_-_R062_6_17_id002_y1973_x3197_mag28.549_snr1.24_R062_norm.fits"

    Parameters
    ----------
    filename : str
        Filename to extract metadata from

    Returns
    -------
    dict
        Dictionary containing extracted metadata (snr, magnitude, id, x, y, band)
    """
    import re

    metadata = {}

    # Extract SNR with regex
    snr_match = re.search(r'snr(\d+\.\d+)', filename)
    metadata['snr'] = float(snr_match.group(1)) if snr_match else None

    # Extract magnitude with regex
    mag_match = re.search(r'mag(\d+\.\d+)', filename)
    metadata['magnitude'] = float(mag_match.group(1)) if mag_match else None

    # Extract ID if present
    id_match = re.search(r'id(\d+)', filename)
    metadata['id'] = int(id_match.group(1)) if id_match else None

    # Extract coordinates if present
    x_match = re.search(r'x(\d+)', filename)
    y_match = re.search(r'y(\d+)', filename)

    if x_match and y_match:
        metadata['x'] = int(x_match.group(1))
        metadata['y'] = int(y_match.group(1))
    else:
        metadata['x'] = None
        metadata['y'] = None

    # Extract band if present
    band_match = re.search(r'_([RYJHFKZ]\d+)_', filename)
    metadata['band'] = band_match.group(1) if band_match else None

    return metadata


def create_summary_report(results, output_folder):
    """Create a summary report of the processing results."""
    # Create summary statistics
    total_processed = results['positive_count'] + results['negative_count']

    # Group by subfolder
    subfolder_stats = {}
    for result in results['results']:
        subfolder = result['subfolder']
        if subfolder not in subfolder_stats:
            subfolder_stats[subfolder] = {'positive': 0, 'negative': 0, 'total': 0}

        if result['prediction'] == 1:
            subfolder_stats[subfolder]['positive'] += 1
        else:
            subfolder_stats[subfolder]['negative'] += 1
        subfolder_stats[subfolder]['total'] += 1

    # Create summary plot
    plt.figure(figsize=(15, 10))

    # Plot 1: Overall summary
    plt.subplot(2, 2, 1)
    categories = ['Positive', 'Negative', 'Failed']
    counts = [results['positive_count'], results['negative_count'], results['failed_count']]
    colors = ['green', 'red', 'orange']

    bars = plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.title('Overall Processing Summary\n(ZScale Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Files')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{count}', ha='center', va='bottom', fontsize=12)

    # Plot 2: Probability distribution
    plt.subplot(2, 2, 2)
    probabilities = [result['probability'] for result in results['results']]
    plt.hist(probabilities, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Probability Score')
    plt.ylabel('Number of Files')
    plt.title('Probability Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()

    # Plot 3: Subfolder breakdown
    plt.subplot(2, 1, 2)
    if subfolder_stats:
        subfolders = list(subfolder_stats.keys())
        positive_counts = [subfolder_stats[sf]['positive'] for sf in subfolders]
        negative_counts = [subfolder_stats[sf]['negative'] for sf in subfolders]

        x = np.arange(len(subfolders))
        width = 0.35

        plt.bar(x - width/2, positive_counts, width, label='Positive', color='green', alpha=0.7)
        plt.bar(x + width/2, negative_counts, width, label='Negative', color='red', alpha=0.7)

        plt.xlabel('Subfolder')
        plt.ylabel('Number of Files')
        plt.title('Predictions by Subfolder', fontsize=14, fontweight='bold')
        plt.xticks(x, subfolders, rotation=45, ha='right')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'processing_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create text summary
    summary_text = f"Processing Summary\n{'='*50}\n"
    summary_text += f"Total processed: {total_processed}\n"
    summary_text += f"Positive detections: {results['positive_count']}\n"
    summary_text += f"Negative detections: {results['negative_count']}\n"
    summary_text += f"Failed: {results['failed_count']}\n"

    # Save text summary
    with open(os.path.join(output_folder, 'processing_summary.txt'), 'w') as f:
        f.write(summary_text)

    print(summary_text)


def main():
    """Main function with variable decision threshold."""
    model_path = 'DenseNet169_best.pth'
    base_cutouts_folder = "cutouts/"
    base_output_folder = "predictions/densenet_predictions_zscale_latest/"

    # Get all subdirectories in the cutouts folder
    cutout_folders = [f for f in os.listdir(base_cutouts_folder)
                      if os.path.isdir(os.path.join(base_cutouts_folder, f))]

    # Process each folder
    for folder_name in cutout_folders:
        input_folder = os.path.join(base_cutouts_folder, folder_name, "fits/")
        output_folder = os.path.join(base_output_folder, folder_name, "")

        # Check if input folder exists
        if os.path.exists(input_folder):
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            print(f"Processing: {input_folder}")
            print(f"Output to: {output_folder}")
        else:
            print(f"Skipping {folder_name}: fits/ subdirectory not found")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Using ZScale normalization (same as training data)")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return

    # Load model
    print(f"Loading DenseNet169 model from {model_path}")
    if model_path == 'DenseNet169_best.pth':
        model = densenet169(num_classes=1)
    elif model_path == 'DenseNet121_best.pth':
        model = densenet121(num_classes=1)
    else:
        print("Warning: model not found")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Process all FITS files
    print("Processing FITS files...")
    results = process_fits_folder(input_folder, output_folder, model, device)

    # Create summary report
    print("Creating summary report...")
    create_summary_report(results, output_folder)


if __name__ == '__main__':
    main()

