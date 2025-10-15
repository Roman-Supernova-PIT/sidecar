"""
User: BCN
Date: Oct 14 2025
Prerequisite: Should have ecsv/<tilename>.ecsv and fits folder should have corresponding fits file.
what this code does: Makes cutouts (normalized) and then saves it in cutouts/<tilename>/fits and png. 
"""

import os
import numpy as np
from astropy.io import ascii, fits
from astropy.visualization import ZScaleInterval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
import json

warnings.filterwarnings('ignore')

try:
    import torch
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        device = torch.device('cuda')
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
except ImportError:
    USE_GPU = False
    device = None
    print("💻 Using CPU")

def normalize_with_zscale(data):
    """Normalize data using ZScale."""
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data[valid_mask])
    normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    normalized[~valid_mask] = 0
    return normalized

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

def save_checkpoint(checkpoint_file, processed_indices):
    """Save progress checkpoint."""
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_indices': list(processed_indices)}, f)

def load_checkpoint(checkpoint_file):
    """Load progress checkpoint."""
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data['processed_indices'])
    return set()

def detect_catalog_type(data):
    """
    Detect which type of catalog this is based on column names.
    
    Returns:
    --------
    catalog_type : str
        'detection' for cleaned_detection_to_transients (X_IMAGE, Y_IMAGE)
        'score' for cleaned_score_detection_to_transients (x_centroid, y_centroid)
    x_col : str
        Name of X coordinate column
    y_col : str
        Name of Y coordinate column
    id_col : str
        Name of ID column
    """
    colnames = data.colnames
    
    # Check for detection catalog (X_IMAGE, Y_IMAGE)
    if 'X_IMAGE' in colnames and 'Y_IMAGE' in colnames:
        id_col = 'NUMBER' if 'NUMBER' in colnames else 'id'
        return 'detection', 'X_IMAGE', 'Y_IMAGE', id_col
    
    # Check for score catalog (x_centroid, y_centroid)
    elif 'x_centroid' in colnames and 'y_centroid' in colnames:
        id_col = 'id' if 'id' in colnames else 'NUMBER'
        return 'score', 'x_centroid', 'y_centroid', id_col
    
    else:
        raise ValueError(f"Unknown catalog type. Columns: {colnames}")

def extract_cutout_with_padding(image_data, center_x, center_y, cutout_size=64, pad_value=0):
    """
    Extract cutout with padding if it goes beyond image boundaries.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Full image
    center_x, center_y : int
        Center coordinates (0-indexed)
    cutout_size : int
        Size of cutout
    pad_value : float
        Value to use for padding
        
    Returns:
    --------
    cutout : numpy.ndarray
        Cutout of exactly cutout_size x cutout_size
    """
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

def process_batch(image_data, batch_data, output_dirs, x_col, y_col, id_col,
                  cutout_size=64, disable_png=False, debug_first=False, 
                  allow_edge_cutouts=True):
    """
    Process a batch of cutouts.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Full FITS image
    batch_data : list
        List of (index, row) tuples
    output_dirs : dict
        Output directories
    x_col, y_col, id_col : str
        Column names for coordinates and ID
    cutout_size : int
        Size of cutouts
    disable_png : bool
        Skip PNG creation
    debug_first : bool
        Print debug info for first few
    allow_edge_cutouts : bool
        Allow edge cutouts with padding
    """
    successful = 0
    skipped = 0
    skip_reasons = {
        'out_of_bounds': 0,
        'wrong_shape': 0,
        'exception': 0,
        'all_padding': 0
    }
    
    half_size = cutout_size // 2
    
    for batch_idx, (idx, row) in enumerate(batch_data):
        try:
            # Extract coordinates based on catalog type
            x_image = float(row[x_col])
            y_image = float(row[y_col])
            
            # Get ID
            try:
                number = int(row[id_col])
            except:
                number = idx  # Fallback to index if ID not available
            
            # Convert to 0-indexed
            # Note: FITS uses 1-indexed, but centroids might already be 0-indexed
            # Check the values to determine
            if x_col in ['X_IMAGE', 'Y_IMAGE']:
                # SExtractor convention: 1-indexed
                x_center = int(x_image - 1)
                y_center = int(y_image - 1)
            else:
                # Centroid convention: typically 0-indexed
                x_center = int(x_image)
                y_center = int(y_image)
            
            # Debug first few
            if debug_first and batch_idx < 3:
                print(f"\n  Debug cutout {batch_idx + 1}:")
                print(f"    {x_col}={x_image:.2f}, {y_col}={y_image:.2f}")
                print(f"    x_center={x_center}, y_center={y_center}")
                print(f"    Image shape: {image_data.shape}")
            
            # Check if completely outside image
            if (x_center < -half_size or x_center >= image_data.shape[1] + half_size or
                y_center < -half_size or y_center >= image_data.shape[0] + half_size):
                if debug_first and batch_idx < 3:
                    print(f"    ✗ Completely outside image!")
                skip_reasons['out_of_bounds'] += 1
                skipped += 1
                continue
            
            if not allow_edge_cutouts:
                # Old behavior: skip edge cutouts
                y_start = y_center - half_size
                y_end = y_center + half_size
                x_start = x_center - half_size
                x_end = x_center + half_size
                
                if (y_start < 0 or y_end > image_data.shape[0] or 
                    x_start < 0 or x_end > image_data.shape[1]):
                    skip_reasons['out_of_bounds'] += 1
                    skipped += 1
                    continue
                
                cutout_raw = image_data[y_start:y_end, x_start:x_end].copy()
            else:
                # New behavior: use padding for edge cutouts
                cutout_raw = extract_cutout_with_padding(
                    image_data, x_center, y_center, cutout_size, pad_value=0
                )
            
            if debug_first and batch_idx < 3:
                print(f"    Cutout shape: {cutout_raw.shape}")
                print(f"    Cutout range: [{np.nanmin(cutout_raw):.2f}, {np.nanmax(cutout_raw):.2f}]")
                print(f"    Non-zero pixels: {np.sum(cutout_raw != 0)}/{cutout_raw.size}")
            
            # Verify size
            if cutout_raw.shape != (cutout_size, cutout_size):
                if debug_first and batch_idx < 3:
                    print(f"    ✗ Wrong shape!")
                skip_reasons['wrong_shape'] += 1
                skipped += 1
                continue
            
            # Skip if cutout is all padding (all zeros)
            if np.all(cutout_raw == 0):
                if debug_first and batch_idx < 3:
                    print(f"    ✗ All padding/zeros!")
                skip_reasons['all_padding'] += 1
                skipped += 1
                continue
            
            # Create NORMALIZED cutout
            cutout_normalized = normalize_cutout_0_1(cutout_raw)
            
            # Filenames with x,y positions
            base_filename = f"cutout_{number:04d}_x{int(x_image)}_y{int(y_image)}"
            
            # Save RAW FITS
            fits_raw_file = output_dirs['fits_raw'] / f"{base_filename}_raw.fits"
            raw_hdu = fits.PrimaryHDU(cutout_raw)
            raw_hdu.header['XCENTER'] = (x_image, f'{x_col} from catalog')
            raw_hdu.header['YCENTER'] = (y_image, f'{y_col} from catalog')
            raw_hdu.header['OBJNUM'] = (number, f'{id_col} from catalog')
            raw_hdu.header['XCOLNAME'] = (x_col, 'X coordinate column name')
            raw_hdu.header['YCOLNAME'] = (y_col, 'Y coordinate column name')
            raw_hdu.header['CUTSIZE'] = (cutout_size, 'Cutout size in pixels')
            raw_hdu.header['ZNORM'] = (False, 'Raw flux values preserved')
            raw_hdu.header['EDGECUT'] = (allow_edge_cutouts, 'Edge cutouts allowed')
            raw_hdu.header['COMMENT'] = 'Raw cutout - flux values preserved'
            raw_hdu.writeto(fits_raw_file, overwrite=True)
            
            # Save NORMALIZED FITS
            fits_norm_file = output_dirs['fits_norm'] / f"{base_filename}_norm.fits"
            norm_hdu = fits.PrimaryHDU(cutout_normalized)
            norm_hdu.header['XCENTER'] = (x_image, f'{x_col} from catalog')
            norm_hdu.header['YCENTER'] = (y_image, f'{y_col} from catalog')
            norm_hdu.header['OBJNUM'] = (number, f'{id_col} from catalog')
            norm_hdu.header['XCOLNAME'] = (x_col, 'X coordinate column name')
            norm_hdu.header['YCOLNAME'] = (y_col, 'Y coordinate column name')
            norm_hdu.header['CUTSIZE'] = (cutout_size, 'Cutout size in pixels')
            norm_hdu.header['ZNORM'] = (True, 'Normalized to 0-1 range')
            norm_hdu.header['NORMTYPE'] = ('MIN_MAX', 'Normalization method')
            norm_hdu.header['EDGECUT'] = (allow_edge_cutouts, 'Edge cutouts allowed')
            norm_hdu.header['COMMENT'] = 'Normalized cutout - 0-1 range for ML'
            norm_hdu.writeto(fits_norm_file, overwrite=True)
            
            # Save PNG
            if not disable_png:
                png_file = output_dirs['png'] / f"{base_filename}.png"
                cutout_display = normalize_with_zscale(cutout_raw)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(cutout_display, origin='lower', cmap='gray')
                
                # Mark center
                ax.plot(cutout_size/2, cutout_size/2, 'r+', markersize=15, markeredgewidth=2)
                ax.set_title(f'Object {number}\nX={x_image:.2f}, Y={y_image:.2f}')
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                plt.colorbar(im, ax=ax, label='Normalized Flux')
                
                plt.tight_layout()
                plt.savefig(png_file, dpi=100, bbox_inches='tight')
                plt.close(fig)
                plt.clf()
            
            if debug_first and batch_idx < 3:
                print(f"    ✓ Success! Saved {base_filename}")
            
            successful += 1
            
            # Clean up
            del cutout_raw, cutout_normalized
            if not disable_png:
                del cutout_display
            
        except Exception as e:
            if debug_first and batch_idx < 3:
                print(f"    ✗ Exception: {e}")
                import traceback
                traceback.print_exc()
            skip_reasons['exception'] += 1
            skipped += 1
            continue
    
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
    
    return successful, skipped, skip_reasons

def determine_fits_basename(ecsv_basename):
    """
    Determine the corresponding FITS filename from ECSV basename.
    
    Handles both:
    - cleaned_detection_to_transients_XXX -> decorr_diff_XXX
    - cleaned_score_detection_to_transients_XXX -> decorr_diff_XXX (or score_XXX if separate)
    """
    if ecsv_basename.startswith('cleaned_detection_to_transients_'):
        return ecsv_basename.replace('cleaned_detection_to_transients_', 'decorr_diff_')
    elif ecsv_basename.startswith('cleaned_score_detection_to_transients_'):
        return ecsv_basename.replace('cleaned_score_detection_to_transients_', 'decorr_diff_')
    else:
        return f"decorr_diff_{ecsv_basename}"

def process_ecsv_and_create_cutouts(ecsv_dir='ecsv', fits_dir='fits', 
                                     output_base='cutouts', cutout_size=64,
                                     batch_size=500, disable_png=False,
                                     resume=True, force_restart=False,
                                     allow_edge_cutouts=True):
    """
    Process ECSV files and create cutouts.
    Automatically detects catalog type and uses appropriate coordinate columns.
    """
    
    Path(output_base).mkdir(parents=True, exist_ok=True)
    ecsv_files = list(Path(ecsv_dir).glob('*.ecsv'))
    
    if not ecsv_files:
        print(f"No ECSV files found in {ecsv_dir}")
        return
    
    print(f"=" * 70)
    print(f"Found {len(ecsv_files)} ECSV file(s)")
    print(f"Batch size: {batch_size} cutouts")
    print(f"PNG creation: {'Disabled' if disable_png else 'Enabled'}")
    print(f"Edge cutouts: {'Allowed (with padding)' if allow_edge_cutouts else 'Skipped'}")
    print(f"=" * 70)
    
    total_cutouts_created = 0
    total_skipped = 0
    all_skip_reasons = {'out_of_bounds': 0, 'wrong_shape': 0, 'exception': 0, 'all_padding': 0}
    
    for file_idx, ecsv_file in enumerate(ecsv_files, 1):
        print(f"\n{'='*70}")
        print(f"[{file_idx}/{len(ecsv_files)}] {ecsv_file.name}")
        print(f"{'='*70}")
        
        try:
            data = ascii.read(str(ecsv_file), format='ecsv')
            print(f"Loaded ECSV with {len(data):,} objects")
            
            # Detect catalog type
            catalog_type, x_col, y_col, id_col = detect_catalog_type(data)
            print(f"Detected catalog type: '{catalog_type}'")
            print(f"Using columns: X={x_col}, Y={y_col}, ID={id_col}")
            
            # Show sample coordinates
            print(f"  Sample coordinates:")
            for i in range(min(3, len(data))):
                obj_id = data[i][id_col] if id_col in data.colnames else i
                print(f"    Object {obj_id}: {x_col}={data[i][x_col]:.2f}, {y_col}={data[i][y_col]:.2f}")
            
        except Exception as e:
            print(f"Error reading: {e}")
            continue
        
        # Determine FITS filename
        ecsv_basename = ecsv_file.stem
        fits_basename = determine_fits_basename(ecsv_basename)
        fits_file = Path(fits_dir) / f"{fits_basename}.fits"
        
        if not fits_file.exists():
            print(f"FITS file not found: {fits_file}")
            continue
        
        # Create output directories
        output_dir = Path(output_base) / fits_basename
        fits_raw_dir = output_dir / 'fits_raw'
        fits_norm_dir = output_dir / 'fits'
        png_dir = output_dir / 'png'
        
        fits_raw_dir.mkdir(parents=True, exist_ok=True)
        fits_norm_dir.mkdir(parents=True, exist_ok=True)
        if not disable_png:
            png_dir.mkdir(parents=True, exist_ok=True)
        
        output_dirs = {'fits_raw': fits_raw_dir, 'fits_norm': fits_norm_dir, 'png': png_dir}
        checkpoint_file = output_dir / f'checkpoint_{catalog_type}.json'
        
        processed_indices = set()
        if force_restart and checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"Deleted checkpoint")
        elif resume:
            processed_indices = load_checkpoint(checkpoint_file)
            if processed_indices:
                print(f"Resuming: {len(processed_indices):,} already processed")
        
        try:
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                if image_data.ndim > 2:
                    image_data = image_data[0] if image_data.ndim == 3 else image_data.squeeze()
                print(f"Loaded FITS: {fits_file.name} (shape: {image_data.shape})")
        except Exception as e:
            print(f"Error reading FITS: {e}")
            continue
        
        successful_cutouts = 0
        skipped_cutouts = 0
        
        remaining_indices = [i for i in range(len(data)) if i not in processed_indices]
        total_batches = (len(remaining_indices) + batch_size - 1) // batch_size
        
        print(f"Processing {len(remaining_indices):,} objects in {total_batches} batches")
        
        for batch_idx in tqdm(range(0, len(remaining_indices), batch_size),
                             desc="Processing batches", unit="batch", ncols=100):
            
            batch_indices = remaining_indices[batch_idx:batch_idx + batch_size]
            batch_data = [(i, data[i]) for i in batch_indices]
            
            debug_first = (batch_idx == 0)
            if debug_first:
                print(f"\n  Debugging first 3 cutouts (catalog type: {catalog_type}):")
            
            batch_successful, batch_skipped, skip_reasons = process_batch(
                image_data, batch_data, output_dirs, x_col, y_col, id_col,
                cutout_size, disable_png, debug_first, allow_edge_cutouts
            )
            
            successful_cutouts += batch_successful
            skipped_cutouts += batch_skipped
            
            for reason, count in skip_reasons.items():
                all_skip_reasons[reason] += count
            
            processed_indices.update(batch_indices)
            if resume and batch_idx % (batch_size * 5) == 0:
                save_checkpoint(checkpoint_file, processed_indices)
            
            gc.collect()
            if USE_GPU:
                torch.cuda.empty_cache()
        
        if resume:
            save_checkpoint(checkpoint_file, processed_indices)
        
        del image_data
        gc.collect()
        
        print(f"\n{'─'*70}")
        print(f"Summary for {ecsv_file.name} ({catalog_type}):")
        print(f"Created: {successful_cutouts:,} cutouts")
        if skipped_cutouts > 0:
            print(f"Skipped: {skipped_cutouts:,}")
        if successful_cutouts + skipped_cutouts > 0:
            print(f"Success: {100*successful_cutouts/(successful_cutouts+skipped_cutouts):.1f}%")
        print(f"{'─'*70}")
        
        total_cutouts_created += successful_cutouts
        total_skipped += skipped_cutouts
    
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Total created: {total_cutouts_created:,}")
    print(f"Total skipped: {total_skipped:,}")
    if total_cutouts_created + total_skipped > 0:
        print(f"Success rate: {100*total_cutouts_created/(total_cutouts_created+total_skipped):.1f}%")
    if total_skipped > 0:
        print(f"\nSkip reasons:")
        for reason, count in all_skip_reasons.items():
            if count > 0:
                print(f"  - {reason}: {count:,}")
    print(f"{'='*70}")

if __name__ == "__main__":
    process_ecsv_and_create_cutouts(
        batch_size=500,
        disable_png=False,
        resume=True,
        force_restart=True,
        allow_edge_cutouts=True
    )
    print("\n Complete!")