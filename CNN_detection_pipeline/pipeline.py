#!/usr/bin/env python3
"""Simple pipeline to process DIA outputs:
1. Extract cutouts from catalogs
2. Run CNN predictions on cutouts

User: BCN
Date: Nov 04 2025

Usage:
    python pipeline.py
"""

import sys
import os


def run_cutout_extraction(dia_out_dir, cutout_size, batch_size, disable_png):
    """Step 1: Extract cutouts"""
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING CUTOUTS")
    print("="*70 + "\n")

    try:
        from make_cutouts import process_ecsv_and_create_cutouts

        process_ecsv_and_create_cutouts(
            dia_out_dir=dia_out_dir,
            cutout_size=cutout_size,
            batch_size=batch_size,
            disable_png=disable_png,
            resume=True,
            force_restart=False,
            allow_edge_cutouts=True
        )

        print("\nCutout extraction completed")
        return True

    except Exception as e:
        print(f"\nError during cutout extraction: {e}")
        return False


def run_cnn_prediction(dia_out_dir, model_path, threshold):
    """Step 2: Run CNN predictions"""
    print("\n" + "="*70)
    print("STEP 2: RUNNING CNN PREDICTIONS")
    print("="*70 + "\n")

    try:
        import torch
        from pathlib import Path
        from CNN_prediction import (
            densenet169, densenet121,
            process_fits_folder_with_threshold,
            create_summary_report
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"Decision threshold: {threshold}\n")

        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False

        # Load model
        print(f"Loading model: {model_path}")
        if 'DenseNet169' in model_path:
            model = densenet169(num_classes=1)
        elif 'DenseNet121' in model_path:
            model = densenet121(num_classes=1)
        else:
            model = densenet169(num_classes=1)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded\n")

        # Find subdirectories
        subdirs = [d for d in Path(dia_out_dir).iterdir() if d.is_dir()]

        if not subdirs:
            print(f"No subdirectories found in {dia_out_dir}")
            return False

        print(f"Found {len(subdirs)} subdirectories\n")

        # Process each subdirectory
        for idx, subdir in enumerate(subdirs, 1):
            print(f"\n[{idx}/{len(subdirs)}] {subdir.name}")

            cutouts_folder = subdir / 'cutouts' / 'fits'

            if not cutouts_folder.exists():
                print("  Skipping - no cutouts folder")
                continue

            output_folder = subdir / 'cnn_detection_results'
            output_folder.mkdir(parents=True, exist_ok=True)

            print("  Processing cutouts...")
            results = process_fits_folder_with_threshold(
                str(cutouts_folder),
                str(output_folder),
                model,
                device,
                threshold
            )

            print("  Creating summary...")
            create_summary_report(results, str(output_folder))
            print("  Done")

        print("\nCNN predictions completed")
        return True

    except Exception as e:
        print(f"\nError during CNN prediction: {e}")
        return False


def main():
    """Main pipeline"""
    print("\n" + "="*70)
    print("DIA PROCESSING PIPELINE")
    print("="*70)
    print("Step 1: Extract cutouts")
    print("Step 2: Run CNN predictions")
    print("="*70 + "\n")

    # Configuration
    DIA_OUT_DIR = '../dia_out_dir'
    CUTOUT_SIZE = 64
    BATCH_SIZE = 500
    DISABLE_PNG = False
    MODEL_PATH = 'DenseNet169_best.pth'
    THRESHOLD = 0.50

    # Run pipeline
    step1_ok = run_cutout_extraction(DIA_OUT_DIR, CUTOUT_SIZE, BATCH_SIZE, DISABLE_PNG)

    if not step1_ok:
        print("\nPipeline failed at Step 1")
        sys.exit(1)

    step2_ok = run_cnn_prediction(DIA_OUT_DIR, MODEL_PATH, THRESHOLD)

    if not step2_ok:
        print("\nPipeline failed at Step 2")
        sys.exit(1)

    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Results in: {DIA_OUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
