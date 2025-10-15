#!/usr/bin/env python3
"""
User: BCN
Date: 14 Oct 2025

Prerequisite: make_cutouts.py should have been executed before this. Such that, cutouts/<tile_name>/fits folder exists.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from tqdm import tqdm
from skimage.transform import resize
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class _DenseLayer(nn.Module):
    """Dense layer implementation following original DenseNet paper"""
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
        """Bottleneck function"""
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
            
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
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
    """Dense block implementation following original DenseNet paper"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    """Transition layer implementation following original DenseNet paper"""
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """DenseNet-BC model implementation"""
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
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
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
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()

def densenet169(**kwargs):
    """DenseNet-169 model"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
def densenet121(**kwargs):
    """DenseNet-121 model"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)

def normalize_with_zscale(data):
    """
    Normalize data using astropy's ZScale algorithm.
    This is the SAME function used in peak finder and injection cutouts scripts.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data array
    
    Returns:
    --------
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
    """Load and preprocess a single FITS file with correct ZScale normalization"""
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
    """Make prediction on a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probability = output.cpu().numpy().item()
        prediction = 1 if probability >= 0.5 else 0
        return prediction, probability

def save_image_with_prediction(image_data, prediction, probability, original_path, output_folder, metadata):
    """
    Save image as PNG with prediction info in filename and metadata overlay.
    
    Parameters:
    -----------
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
    
    Returns:
    --------
    str
        Path to saved PNG file
    """
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    
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
    plt.gca().spines['top'].set_color(border_color)
    plt.gca().spines['bottom'].set_color(border_color)
    plt.gca().spines['left'].set_color(border_color)
    plt.gca().spines['right'].set_color(border_color)
    plt.gca().spines['top'].set_linewidth(4)
    plt.gca().spines['bottom'].set_linewidth(4)
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def process_fits_folder(input_folder, output_folder, model, device):
    """
    Process all FITS files in the input folder and save results.
    
    Parameters:
    -----------
    input_folder : str
        Folder containing FITS files to process
    output_folder : str
        Folder to save results
    model : torch.nn.Module
        Trained DenseNet model
    device : torch.device
        Device to run model on (CPU or CUDA)
    
    Returns:
    --------
    dict
        Dictionary containing processing results
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    
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
    """
    Extract SNR and magnitude values from filename.
    Example: "psf_injection_diff_R062_41205_12_-_R062_6_17_id002_y1973_x3197_mag28.549_snr1.24_R062_norm.fits"
    
    Parameters:
    -----------
    filename : str
        Filename to extract metadata from
        
    Returns:
    --------
    dict
        Dictionary containing extracted metadata (snr, magnitude, id, x, y, band)
    """
    import re
    
    metadata = {}
    
    # Extract SNR with regex
    snr_match = re.search(r'snr(\d+\.\d+)', filename)
    if snr_match:
        metadata['snr'] = float(snr_match.group(1))
    else:
        metadata['snr'] = None
    
    # Extract magnitude with regex
    mag_match = re.search(r'mag(\d+\.\d+)', filename)
    if mag_match:
        metadata['magnitude'] = float(mag_match.group(1))
    else:
        metadata['magnitude'] = None
    
    # Extract ID if present
    id_match = re.search(r'id(\d+)', filename)
    if id_match:
        metadata['id'] = int(id_match.group(1))
    else:
        metadata['id'] = None
    
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
    if band_match:
        metadata['band'] = band_match.group(1)
    else:
        metadata['band'] = None
    
    return metadata

def create_snr_magnitude_analysis(results, output_folder):
    """
    Create plots analyzing detection results as a function of SNR and magnitude.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results with all detection results
    output_folder : str
        Path to save the output plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.interpolate import griddata
    
    # Filter out results with missing SNR or magnitude values
    valid_results = [r for r in results['results'] if r['snr'] is not None and r['magnitude'] is not None]
    
    if not valid_results:
        print("No valid SNR/magnitude data found in filenames.")
        return
    
    # Extract data
    snrs = [r['snr'] for r in valid_results]
    magnitudes = [r['magnitude'] for r in valid_results]
    probabilities = [r['probability'] for r in valid_results]
    predictions = [r['prediction'] for r in valid_results]
    
    print(f"Analyzing {len(valid_results)} results with valid SNR and magnitude data")
    print(f"SNR range: {min(snrs):.2f} to {max(snrs):.2f}")
    print(f"Magnitude range: {min(magnitudes):.2f} to {max(magnitudes):.2f}")
    
    # Create figure for SNR analysis
    plt.figure(figsize=(15, 10))
    
    # Plot 1: SNR vs. Probability
    plt.subplot(2, 2, 1)
    plt.scatter(snrs, probabilities, c=predictions, cmap='coolwarm', alpha=0.7, s=50, edgecolor='black')
    cbar = plt.colorbar(label='Prediction (1=Positive, 0=Negative)', ticks=[0, 1])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Prediction (1=Positive, 0=Negative)', fontsize=16)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.ylabel('Detection Probability', fontsize=16)
    plt.title('SNR vs. Detection Probability', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    
    # Plot 2: SNR distribution by prediction
    plt.subplot(2, 2, 2)
    # Create separate lists for positive and negative detections
    pos_snrs = [r['snr'] for r in valid_results if r['prediction'] == 1]
    neg_snrs = [r['snr'] for r in valid_results if r['prediction'] == 0]
    
    bins = np.linspace(min(snrs), max(snrs), 20)
    plt.hist([pos_snrs, neg_snrs], bins=bins, alpha=0.7, color=['green', 'red'], 
             label=['Positive', 'Negative'], stacked=True, edgecolor='black')
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('SNR Distribution by Prediction', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Magnitude vs. Probability
    plt.subplot(2, 2, 3)
    plt.scatter(magnitudes, probabilities, c=predictions, cmap='coolwarm', alpha=0.7, s=50, edgecolor='black')
    cbar = plt.colorbar(label='Prediction (1=Positive, 0=Negative)', ticks=[0, 1])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Prediction (1=Positive, 0=Negative)', fontsize=16)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Magnitude', fontsize=16)
    plt.ylabel('Detection Probability', fontsize=16)
    plt.title('Magnitude vs. Detection Probability', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    
    # Plot 4: Magnitude distribution by prediction
    plt.subplot(2, 2, 4)
    # Create separate lists for positive and negative detections
    pos_mags = [r['magnitude'] for r in valid_results if r['prediction'] == 1]
    neg_mags = [r['magnitude'] for r in valid_results if r['prediction'] == 0]
    
    bins = np.linspace(min(magnitudes), max(magnitudes), 20)
    plt.hist([pos_mags, neg_mags], bins=bins, alpha=0.7, color=['green', 'red'], 
             label=['Positive', 'Negative'], stacked=True, edgecolor='black')
    plt.xlabel('Magnitude', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Magnitude Distribution by Prediction', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'snr_magnitude_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detection rate plot for SNR bins
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Detection rate by SNR bins
    plt.subplot(1, 2, 1)
    snr_bins = np.linspace(min(snrs), max(snrs), 10)
    snr_bin_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
    
    detection_rates = []
    detection_errors = []
    bin_counts = []
    
    for i in range(len(snr_bins) - 1):
        # Find results in this bin
        bin_results = [r for r in valid_results 
                      if r['snr'] >= snr_bins[i] and r['snr'] < snr_bins[i+1]]
        
        if len(bin_results) > 0:
            # Calculate detection rate
            detection_count = sum(1 for r in bin_results if r['prediction'] == 1)
            rate = detection_count / len(bin_results)
            
            # Calculate error (95% confidence interval)
            # Using Wilson score interval for small sample sizes
            z = 1.96  # 95% confidence
            n = len(bin_results)
            error = z * np.sqrt((rate * (1 - rate) + z**2/(4*n)) / n) / (1 + z**2/n)
            
            detection_rates.append(rate)
            detection_errors.append(error)
            bin_counts.append(len(bin_results))
        else:
            detection_rates.append(0)
            detection_errors.append(0)
            bin_counts.append(0)
    
    # Plot detection rate
    plt.errorbar(snr_bin_centers, detection_rates, yerr=detection_errors, 
                 fmt='o-', capsize=5, color='blue', linewidth=2, markersize=8)
    
    # Add count labels
    for i, (x, y, count) in enumerate(zip(snr_bin_centers, detection_rates, bin_counts)):
        plt.annotate(f'n={count}', (x, y + 0.05), ha='center', va='bottom', fontsize=16)
    
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.ylabel('Detection Rate', fontsize=16)
    plt.title('Detection Rate by SNR', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Detection rate by magnitude bins
    plt.subplot(1, 2, 2)
    mag_bins = np.linspace(min(magnitudes), max(magnitudes), 10)
    mag_bin_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
    
    detection_rates = []
    detection_errors = []
    bin_counts = []
    
    for i in range(len(mag_bins) - 1):
        # Find results in this bin
        bin_results = [r for r in valid_results 
                      if r['magnitude'] >= mag_bins[i] and r['magnitude'] < mag_bins[i+1]]
        
        if len(bin_results) > 0:
            # Calculate detection rate
            detection_count = sum(1 for r in bin_results if r['prediction'] == 1)
            rate = detection_count / len(bin_results)
            
            # Calculate error (95% confidence interval)
            z = 1.96  # 95% confidence
            n = len(bin_results)
            error = z * np.sqrt((rate * (1 - rate) + z**2/(4*n)) / n) / (1 + z**2/n)
            
            detection_rates.append(rate)
            detection_errors.append(error)
            bin_counts.append(len(bin_results))
        else:
            detection_rates.append(0)
            detection_errors.append(0)
            bin_counts.append(0)
    
    # Plot detection rate
    plt.errorbar(mag_bin_centers, detection_rates, yerr=detection_errors, 
                 fmt='o-', capsize=5, color='green', linewidth=2, markersize=8)
    
    # Add count labels
    for i, (x, y, count) in enumerate(zip(mag_bin_centers, detection_rates, bin_counts)):
        plt.annotate(f'n={count}', (x, y + 0.05), ha='center', va='bottom', fontsize=16)
    
    plt.xlabel('Magnitude', fontsize=16)
    plt.ylabel('Detection Rate', fontsize=16)
    plt.title('Detection Rate by Magnitude', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'detection_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 2D heatmap of detection rates by SNR and magnitude
    plt.figure(figsize=(12, 10))
    
    # Create 2D bins
    snr_bins = np.linspace(min(snrs), max(snrs), 8)  # Fewer bins to ensure enough samples
    mag_bins = np.linspace(min(magnitudes), max(magnitudes), 8)
    
    # Initialize detection rate matrix
    detection_rates = np.zeros((len(snr_bins)-1, len(mag_bins)-1))
    sample_counts = np.zeros((len(snr_bins)-1, len(mag_bins)-1))
    
    # Calculate detection rates for each bin
    for i in range(len(snr_bins) - 1):
        for j in range(len(mag_bins) - 1):
            # Find results in this bin
            bin_results = [r for r in valid_results 
                          if r['snr'] >= snr_bins[i] and r['snr'] < snr_bins[i+1]
                          and r['magnitude'] >= mag_bins[j] and r['magnitude'] < mag_bins[j+1]]
            
            if len(bin_results) > 0:
                # Calculate detection rate
                detection_count = sum(1 for r in bin_results if r['prediction'] == 1)
                rate = detection_count / len(bin_results)
                
                detection_rates[i, j] = rate
                sample_counts[i, j] = len(bin_results)
            else:
                detection_rates[i, j] = np.nan  # No data
                sample_counts[i, j] = 0
    
    # Create heatmap with labeled axes
    snr_bin_centers = [(snr_bins[i] + snr_bins[i+1]) / 2 for i in range(len(snr_bins) - 1)]
    mag_bin_centers = [(mag_bins[i] + mag_bins[i+1]) / 2 for i in range(len(mag_bins) - 1)]
    
    plt.imshow(detection_rates.T, origin='lower', aspect='auto', 
               extent=[min(snrs), max(snrs), min(magnitudes), max(magnitudes)],
               cmap='viridis', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(label='Detection Rate')
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Detection Rate', fontsize=16)
    
    # Add sample count annotations
    for i in range(len(snr_bins) - 1):
        for j in range(len(mag_bins) - 1):
            if not np.isnan(detection_rates[i, j]):
                plt.text(snr_bin_centers[i], mag_bin_centers[j], f'n={int(sample_counts[i, j])}',
                         ha='center', va='center', color='white' if detection_rates[i, j] > 0.5 else 'black',
                         fontsize=16, fontweight='bold')
    
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.ylabel('Magnitude', fontsize=16)
    plt.title('Detection Rate by SNR and Magnitude', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'detection_rate_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of SNR vs Magnitude colored by detection
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = plt.scatter(snrs, magnitudes, c=predictions, cmap='coolwarm', 
                          alpha=0.7, s=80, edgecolor='black')
    
    # Add colorbar
    cbar = plt.colorbar(label='Detection (1=Positive, 0=Negative)', ticks=[0, 1])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Detection (1=Positive, 0=Negative)', fontsize=16)
    
    # Add contour lines showing probability
    try:
        # Create grid for contour
        snr_grid = np.linspace(min(snrs), max(snrs), 50)
        mag_grid = np.linspace(min(magnitudes), max(magnitudes), 50)
        SNR, MAG = np.meshgrid(snr_grid, mag_grid)
        
        # Use Gaussian kernel regression to smooth probabilities
        PROB = griddata((snrs, magnitudes), probabilities, (SNR, MAG), method='cubic', fill_value=0.5)
        
        # Plot contours
        contour = plt.contour(SNR, MAG, PROB, levels=[0.3, 0.5, 0.7], colors='black', linewidths=2)
        plt.clabel(contour, inline=True, fontsize=16, fmt='%.1f')
    except Exception as e:
        # Skip contours if interpolation fails (e.g., not enough data points)
        print(f"Could not create probability contours: {e}")
        print("Not enough data points or poor distribution.")
    
    plt.xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
    plt.ylabel('Magnitude', fontsize=16)
    plt.title('SNR vs Magnitude with Detection Results', fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'snr_vs_magnitude_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a combined plot of probability, SNR, and magnitude
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot
        scatter = ax.scatter(snrs, magnitudes, probabilities, c=probabilities, 
                            cmap='viridis', alpha=0.7, s=80, edgecolor='black')
        
        # Add colorbar
        cbar = fig.colorbar(scatter, label='Detection Probability')
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Detection Probability', fontsize=16)
        
        # Add decision threshold plane
        x_plane = np.linspace(min(snrs), max(snrs), 10)
        y_plane = np.linspace(min(magnitudes), max(magnitudes), 10)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.ones_like(X_plane) * 0.5  # Decision threshold at 0.5
        
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='red', alpha=0.2)
        
        ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=16)
        ax.set_ylabel('Magnitude', fontsize=16)
        ax.set_zlabel('Detection Probability', fontsize=16)
        ax.set_title('3D Visualization of Detection Probability', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, '3d_probability_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not create 3D visualization: {e}")
    
    print(f"SNR and magnitude analysis plots saved to {output_folder}")
    return



def create_summary_report(results, output_folder):
    """Create a summary report of the processing results"""
    
    # Create summary statistics
    total_processed = results['positive_count'] + results['negative_count']
    positive_rate = results['positive_count'] / total_processed if total_processed > 0 else 0
    
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
    
    
    
    for subfolder, stats in subfolder_stats.items():
        pos_rate = stats['positive'] / stats['total'] if stats['total'] > 0 else 0
        summary_text += f"\n{subfolder}:"
        summary_text += f"\n  Total: {stats['total']}"
        summary_text += f"\n  Positive: {stats['positive']} ({pos_rate:.2%})"
        summary_text += f"\n  Negative: {stats['negative']} ({1-pos_rate:.2%})"
    
    # Save text summary
    with open(os.path.join(output_folder, 'processing_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print(summary_text)

# Add these functions to your existing script:

def predict_single_image_with_threshold(model, image_tensor, device, threshold=0.5):
    """Make prediction on a single image with custom threshold"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        probability = output.cpu().numpy().item()
        prediction = 1 if probability >= threshold else 0
        return prediction, probability

def process_fits_folder_with_threshold(input_folder, output_folder, model, device, threshold=0.5):
    """
    Process all FITS files with custom decision threshold
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    
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
    print(f"Using decision threshold: {threshold}")
    
    # Process each file
    results = {
        'positive_count': 0,
        'negative_count': 0,
        'failed_count': 0,
        'results': [],
        'threshold': threshold
    }
    
    for fits_path in tqdm(fits_files, desc=f"Processing FITS files (threshold={threshold})"):
        # Extract metadata from filename
        filename = os.path.basename(fits_path)
        metadata = extract_metadata_from_filename(filename)
        
        # Load and preprocess with ZScale normalization
        image_tensor, image_data = load_and_preprocess_fits(fits_path)
        
        if image_tensor is None:
            results['failed_count'] += 1
            continue
        
        # Make prediction with custom threshold
        prediction, probability = predict_single_image_with_threshold(model, image_tensor, device, threshold)
        
        # Choose output folder based on prediction
        target_folder = positive_folder if prediction == 1 else negative_folder
        
        # Save image with prediction
        output_path = save_image_with_prediction_threshold(
            image_data, prediction, probability, fits_path, target_folder, metadata, threshold
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

def save_image_with_prediction_threshold(image_data, prediction, probability, original_path, output_folder, metadata, threshold=0.5):
    """
    Save image as PNG with prediction info including threshold
    """
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    
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
    
    # Create title with metadata and threshold
    title = f'Prediction: {pred_label} (threshold={threshold})\nProbability: {probability:.4f}'
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
    plt.gca().spines['top'].set_color(border_color)
    plt.gca().spines['bottom'].set_color(border_color)
    plt.gca().spines['left'].set_color(border_color)
    plt.gca().spines['right'].set_color(border_color)
    plt.gca().spines['top'].set_linewidth(4)
    plt.gca().spines['bottom'].set_linewidth(4)
    plt.gca().spines['left'].set_linewidth(4)
    plt.gca().spines['right'].set_linewidth(4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_roc_curve_analysis(results, output_folder, ground_truth_labels=None):
    """
    Create ROC curve analysis with different thresholds
    
    Parameters:
    -----------
    results : dict
        Results from processing with probabilities
    output_folder : str
        Output folder for plots
    ground_truth_labels : list, optional
        True labels (1 for positive, 0 for negative). If None, uses SNR/magnitude heuristics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import os
    
    # Extract probabilities
    probabilities = [r['probability'] for r in results['results']]
    
    # If no ground truth provided, create heuristic labels based on SNR and magnitude
    if ground_truth_labels is None:
        print("No ground truth provided. Creating heuristic labels based on SNR and magnitude...")
        
        
        ground_truth_labels = []
        for r in results['results']:
            snr = r['snr'] if r['snr'] is not None else 0
            mag = r['magnitude'] if r['magnitude'] is not None else 30
            
            # Heuristic: SNR > 3 AND magnitude < 28 = positive
            if snr > 3 and mag < 28:
                ground_truth_labels.append(1)
            else:
                ground_truth_labels.append(0)
        
        print(f"Created heuristic labels: {sum(ground_truth_labels)} positives, {len(ground_truth_labels) - sum(ground_truth_labels)} negatives")
        print("Note: These are heuristic labels. For accurate ROC analysis, provide true ground truth labels.")
    
    # Create ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Create Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth_labels, probabilities)
    pr_auc = auc(recall, precision)
    
    # Create comprehensive ROC analysis plot
    plt.figure(figsize=(20, 15))
    
    # Plot 1: ROC Curve
    plt.subplot(3, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    # Mark specific thresholds
    threshold_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in threshold_points:
        # Find closest threshold
        idx = np.argmin(np.abs(thresholds - thresh))
        plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8)
        plt.annotate(f'{thresh}', (fpr[idx], tpr[idx]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve with Threshold Markers', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    plt.subplot(3, 3, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Mark specific thresholds on PR curve
    for thresh in threshold_points:
        if thresh < max(pr_thresholds):
            idx = np.argmin(np.abs(pr_thresholds - thresh))
            plt.plot(recall[idx], precision[idx], 'ro', markersize=8)
            plt.annotate(f'{thresh}', (recall[idx], precision[idx]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Threshold vs Metrics
    plt.subplot(3, 3, 3)
    
    # Calculate metrics for different thresholds
    test_thresholds = np.linspace(0.1, 0.9, 17)
    metrics = []
    
    for thresh in test_thresholds:
        predictions = [1 if p >= thresh else 0 for p in probabilities]
        
        tp = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and ground_truth_labels[i] == 1])
        fp = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and ground_truth_labels[i] == 0])
        tn = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and ground_truth_labels[i] == 0])
        fn = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and ground_truth_labels[i] == 1])
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision_val * sensitivity) / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        
        metrics.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision_val,
            'f1': f1,
            'accuracy': accuracy
        })
    
    # Plot metrics vs threshold
    threshs = [m['threshold'] for m in metrics]
    plt.plot(threshs, [m['sensitivity'] for m in metrics], 'o-', label='Sensitivity (TPR)', linewidth=2)
    plt.plot(threshs, [m['specificity'] for m in metrics], 's-', label='Specificity (TNR)', linewidth=2)
    plt.plot(threshs, [m['precision'] for m in metrics], '^-', label='Precision', linewidth=2)
    plt.plot(threshs, [m['f1'] for m in metrics], 'd-', label='F1 Score', linewidth=2)
    plt.plot(threshs, [m['accuracy'] for m in metrics], 'v-', label='Accuracy', linewidth=2)
    
    plt.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Threshold=0.3')
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Threshold=0.5')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Detection Rate by SNR (different thresholds)
    plt.subplot(3, 3, 4)
    
    # Filter results with valid SNR
    snr_results = [r for r in results['results'] if r['snr'] is not None]
    if snr_results:
        snrs = [r['snr'] for r in snr_results]
        snr_probs = [r['probability'] for r in snr_results]
        
        snr_bins = np.linspace(min(snrs), max(snrs), 8)
        snr_centers = (snr_bins[:-1] + snr_bins[1:]) / 2
        
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            detection_rates = []
            for i in range(len(snr_bins) - 1):
                bin_probs = [snr_probs[j] for j in range(len(snr_probs)) 
                           if snr_bins[i] <= snrs[j] < snr_bins[i+1]]
                if bin_probs:
                    rate = sum(1 for p in bin_probs if p >= thresh) / len(bin_probs)
                    detection_rates.append(rate)
                else:
                    detection_rates.append(0)
            
            plt.plot(snr_centers, detection_rates, 'o-', label=f'Threshold={thresh}', linewidth=2)
    
    plt.xlabel('SNR', fontsize=12)
    plt.ylabel('Detection Rate', fontsize=12)
    plt.title('Detection Rate by SNR (Different Thresholds)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Detection Rate by Magnitude (different thresholds)
    plt.subplot(3, 3, 5)
    
    # Filter results with valid magnitude
    mag_results = [r for r in results['results'] if r['magnitude'] is not None]
    if mag_results:
        mags = [r['magnitude'] for r in mag_results]
        mag_probs = [r['probability'] for r in mag_results]
        
        mag_bins = np.linspace(min(mags), max(mags), 8)
        mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
        
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            detection_rates = []
            for i in range(len(mag_bins) - 1):
                bin_probs = [mag_probs[j] for j in range(len(mag_probs)) 
                           if mag_bins[i] <= mags[j] < mag_bins[i+1]]
                if bin_probs:
                    rate = sum(1 for p in bin_probs if p >= thresh) / len(bin_probs)
                    detection_rates.append(rate)
                else:
                    detection_rates.append(0)
            
            plt.plot(mag_centers, detection_rates, 'o-', label=f'Threshold={thresh}', linewidth=2)
    
    plt.xlabel('Magnitude', fontsize=12)
    plt.ylabel('Detection Rate', fontsize=12)
    plt.title('Detection Rate by Magnitude (Different Thresholds)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Confusion Matrix for threshold=0.3
    plt.subplot(3, 3, 6)
    
    thresh_03_predictions = [1 if p >= 0.3 else 0 for p in probabilities]
    tp = sum([1 for i in range(len(thresh_03_predictions)) if thresh_03_predictions[i] == 1 and ground_truth_labels[i] == 1])
    fp = sum([1 for i in range(len(thresh_03_predictions)) if thresh_03_predictions[i] == 1 and ground_truth_labels[i] == 0])
    tn = sum([1 for i in range(len(thresh_03_predictions)) if thresh_03_predictions[i] == 0 and ground_truth_labels[i] == 0])
    fn = sum([1 for i in range(len(thresh_03_predictions)) if thresh_03_predictions[i] == 0 and ground_truth_labels[i] == 1])
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Threshold=0.3)', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, confusion_matrix[i, j], 
                    horizontalalignment="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Plot 7: Probability distribution
    plt.subplot(3, 3, 7)
    
    pos_probs = [probabilities[i] for i in range(len(probabilities)) if ground_truth_labels[i] == 1]
    neg_probs = [probabilities[i] for i in range(len(probabilities)) if ground_truth_labels[i] == 0]
    
    plt.hist(neg_probs, bins=20, alpha=0.7, color='red', label='True Negatives', density=True)
    plt.hist(pos_probs, bins=20, alpha=0.7, color='green', label='True Positives', density=True)
    
    plt.axvline(0.3, color='blue', linestyle='--', linewidth=2, label='Threshold=0.3')
    plt.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Threshold=0.5')
    
    plt.xlabel('Probability Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Probability Distribution by True Label', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Optimal threshold recommendation
    plt.subplot(3, 3, 8)
    
    # Find optimal thresholds using different criteria
    optimal_f1_idx = np.argmax([m['f1'] for m in metrics])
    optimal_f1_thresh = metrics[optimal_f1_idx]['threshold']
    
    # Youden's index (sensitivity + specificity - 1)
    youden_scores = [m['sensitivity'] + m['specificity'] - 1 for m in metrics]
    optimal_youden_idx = np.argmax(youden_scores)
    optimal_youden_thresh = metrics[optimal_youden_idx]['threshold']
    
    plt.plot(threshs, [m['f1'] for m in metrics], 'o-', label='F1 Score', linewidth=2)
    plt.plot(threshs, youden_scores, 's-', label="Youden's Index", linewidth=2)
    
    plt.axvline(optimal_f1_thresh, color='blue', linestyle='--', linewidth=2, 
                label=f'Optimal F1: {optimal_f1_thresh:.2f}')
    plt.axvline(optimal_youden_thresh, color='orange', linestyle='--', linewidth=2, 
                label=f'Optimal Youden: {optimal_youden_thresh:.2f}')
    plt.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Your choice: 0.3')
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Optimal Threshold Selection', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Summary table
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary table for key thresholds
    key_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    summary_data = []
    
    for thresh in key_thresholds:
        # Find closest metric
        closest_idx = np.argmin([abs(m['threshold'] - thresh) for m in metrics])
        m = metrics[closest_idx]
        summary_data.append([
            f"{thresh:.1f}",
            f"{m['sensitivity']:.3f}",
            f"{m['specificity']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['f1']:.3f}"
        ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'F1'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'roc_threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
    
    return metrics, optimal_f1_thresh, optimal_youden_thresh



# Updated main function
def main():
    """
    Main function with variable decision threshold and ROC analysis
    """
    import os
    import torch
    
    DECISION_THRESHOLD = 0.50  # <-- Set your desired threshold here
    
    import os

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
    print(f" Using device: {device}")
    print(f" Using decision threshold: {DECISION_THRESHOLD}")
    print(f" Using ZScale normalization (same as training data)")
    
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
        print("Warning model not found")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f" Model loaded successfully")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Process all FITS files with custom threshold
    print(f"Processing FITS files with threshold {DECISION_THRESHOLD}")
    results = process_fits_folder_with_threshold(input_folder, output_folder, model, device, DECISION_THRESHOLD)
    
    # Create summary report
    print(f"Creating summary report...")
    create_summary_report(results, output_folder)
    
    # Create SNR and magnitude analysis plots
    #print(f"Creating SNR and magnitude analysis plots...")
    #create_snr_magnitude_analysis(results, output_folder)
    
    # Create ROC curve and threshold analysis
    #print(f"Creating ROC curve and threshold analysis...")
    #metrics, optimal_f1, optimal_youden = create_roc_curve_analysis(results, output_folder)
    
    

if __name__ == '__main__':
    main()

