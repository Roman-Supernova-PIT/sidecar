"""Real/bogus classification of difference-image detections.

This module scores each detection in a sidecar detection catalog as a real
astrophysical transient or a bogus detection (subtraction artefact, noise
spike, residual of a bright source, ...).

The entry point is :func:`process_single_catalog_ensemble`. It

1. loads the 24-model CNN ensemble published on Hugging Face as
   ``Roman-Supernova-PIT/transient-real-bogus`` (6 architecture families x 4
   independently trained members) from ``model_dir``
   (``photometry.sidecar.real_bogus.model_dir`` in the config). The models
   are not downloaded during a run: fetch them once beforehand with
   ``python -m sidecar.classifier --download-models`` (see :func:`main`);
2. reads the detection catalog (ECSV or Parquet) and the difference image (FITS) the
   detections were measured on;
3. for every detection, makes a 64x64 cutout centred on its position, in
   memory, and scores it with every model;
4. writes the catalog out (Parquet or ECSV, like the rest of the pipeline) with per-family scores, the ensemble
   score, its spread, and a "real"/"bogus"/"invalid" label added as columns.

It is called from ``sidecar/pipeline.py`` (``Detection.run_one_subtraction``)
after source detection and optional known-star rejection, and needs no truth
catalog, so it runs on real data as well as simulations.

Preprocessing must match training exactly: the whole difference image is
z-scaled to [0, 1], a sub-pixel bilinear cutout is taken from it, and that
cutout is z-scaled to [0, 1] again (see :func:`make_model_cutout`).

A second, 3-channel ensemble (``Roman-Supernova-PIT/Real-Bogus-3ch``) scores
(science, template, diff) cutouts instead of the difference alone; see
:func:`process_single_catalog_ensemble_3ch`. Each channel gets the same
preprocessing as above.
"""

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from snappl.config import Config
from snappl.logger import SNLogger

from sidecar.catalog_io import OUTPUT_FORMATS, read_catalog, write_catalog


# ============================================================================
# DENSENET-169 (architecture of the "DenseNet169" ensemble family)
# ============================================================================

class _DenseLayer(nn.Module):
    """One DenseNet-BC layer: BN-ReLU-Conv1x1 bottleneck, then BN-ReLU-Conv3x3.

    Each layer receives the concatenation of all previous feature maps in its
    dense block and produces ``growth_rate`` new feature maps, following
    Huang et al. (2017), "Densely Connected Convolutional Networks".

    Parameters
    ----------
    num_input_features : int
        Number of channels in the concatenated input feature maps.
    growth_rate : int
        Number of feature maps this layer adds (k in the paper).
    bn_size : int
        Bottleneck width multiplier: the 1x1 convolution outputs
        ``bn_size * growth_rate`` channels.
    drop_rate : float
        Dropout probability applied to the new feature maps during training.
        0 disables dropout.
    """

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
        """Apply the bottleneck (BN-ReLU-Conv1x1) to the concatenated inputs.

        Parameters
        ----------
        inputs : torch.Tensor or list of torch.Tensor
            A single feature-map tensor, or a list of feature maps that are
            concatenated along the channel axis first.

        Returns
        -------
        torch.Tensor
            Bottleneck output with ``bn_size * growth_rate`` channels.
        """
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        """Compute this layer's ``growth_rate`` new feature maps.

        Parameters
        ----------
        input : torch.Tensor or list of torch.Tensor
            Feature map(s) from the block input and all preceding layers.

        Returns
        -------
        torch.Tensor
            New feature maps, shape (batch, growth_rate, H, W).
        """
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
    """A DenseNet dense block: ``num_layers`` densely connected layers.

    Every layer sees the block input plus the outputs of all earlier layers,
    and the block returns all of them concatenated, so the output has
    ``num_input_features + num_layers * growth_rate`` channels.

    Parameters
    ----------
    num_layers : int
        Number of :class:`_DenseLayer` in the block.
    num_input_features : int
        Channels entering the block.
    bn_size : int
        Bottleneck width multiplier passed to each layer.
    growth_rate : int
        Feature maps added by each layer.
    drop_rate : float
        Dropout probability passed to each layer.
    """

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
        """Run all layers and concatenate their outputs with the input.

        Parameters
        ----------
        init_features : torch.Tensor
            Block input, shape (batch, num_input_features, H, W).

        Returns
        -------
        torch.Tensor
            Shape (batch, num_input_features + num_layers * growth_rate, H, W).
        """
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """Transition between dense blocks: BN-ReLU-Conv1x1, then 2x2 average pool.

    Reduces the number of channels (DenseNet-BC compression) and halves the
    spatial size.

    Parameters
    ----------
    num_input_features : int
        Channels entering the transition.
    num_output_features : int
        Channels after the 1x1 convolution.
    """

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """DenseNet-BC classifier returning a real/bogus probability.

    Stem (7x7 conv, stride 2, then 3x3 max-pool), dense blocks separated by
    transitions that halve the channel count, final batch norm, global average
    pooling, and a linear layer followed by a sigmoid.

    The parameter layout matches the checkpoints of the "DenseNet169" family in
    the Hugging Face ensemble, so their ``model_state_dict`` loads directly.

    Parameters
    ----------
    growth_rate : int
        Feature maps added by each dense layer.
    block_config : tuple of int
        Number of layers in each dense block.
    num_init_features : int
        Channels produced by the stem convolution.
    bn_size : int
        Bottleneck width multiplier.
    drop_rate : float
        Dropout probability inside dense layers.
    num_classes : int
        Number of outputs. The ensemble uses 1 (probability of "real").

    Notes
    -----
    Input is a 3-channel image tensor of shape (batch, 3, H, W); the classifier
    feeds the same single-channel cutout in all 3 channels (see
    :func:`to_model_input`).
    """

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
        """Probability that each input cutout is a real transient.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3, H, W), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Sigmoid output in [0, 1]. ``squeeze()`` removes all size-1
            dimensions, so a batch of one gives a 0-d tensor and a batch of N
            gives shape (N,).
        """
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return torch.sigmoid(out).squeeze()


def densenet169(**kwargs):
    """Build a DenseNet-169 (growth rate 32, blocks of 6, 12, 32, 32 layers).

    Parameters
    ----------
    **kwargs
        Passed to :class:`DenseNet`, e.g. ``num_classes=1``.

    Returns
    -------
    DenseNet
        Randomly initialised model; load trained weights with
        ``load_state_dict``.
    """
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)


# ============================================================================
# CATALOG HANDLING
# ============================================================================


def detect_catalog_type(data):
    """Work out which columns of a detection catalog hold the positions.

    Two catalog layouts are supported, checked in this order:

    * sidecar score-image detection catalogs (``score_detection_*.ecsv`` and
      ``cleaned_score_detection_*.ecsv``), with columns such as ``id, x_peak,
      y_peak, peak_value, x_centroid, y_centroid, ra, dec``. Positions are
      taken from ``x_centroid``/``y_centroid`` (0-indexed, sub-pixel).
    * Source Extractor catalogs, with ``X_IMAGE``/``Y_IMAGE`` (1-indexed).

    Parameters
    ----------
    data : astropy.table.Table
        The detection catalog.

    Returns
    -------
    catalog_type : str
        ``"score"`` for x_centroid/y_centroid catalogs, ``"detection"`` for
        Source Extractor catalogs.
    x_col, y_col : str
        Names of the x and y position columns.
    id_col : str
        Name of the identifier column: ``"id"`` or ``"NUMBER"``, whichever the
        catalog type prefers. It is not checked to exist.

    Raises
    ------
    ValueError
        If the catalog has neither pair of position columns.
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


# ============================================================================
# ENSEMBLE (Roman-Supernova-PIT/transient-real-bogus on Hugging Face)
# 6 architecture families x 4 members = 24 models.
#
# IMPORTANT: this ensemble was trained with sub-pixel bilinear cutouts and a
# DOUBLE z-scale normalization (full image, then cutout again) -- see
# normalize_with_zscale/create_cutout below. Changing either convention
# will silently degrade predictions.
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
    """Rescale an image to [0, 1] using IRAF-style z-scale limits.

    The display limits (vmin, vmax) are computed with astropy's
    ``ZScaleInterval`` (default settings: 1000 samples, contrast 0.25) from
    the finite pixels only. Pixels are then mapped linearly so that vmin -> 0
    and vmax -> 1, and clipped to [0, 1]. Z-scale ignores the extreme tails,
    so bright sources saturate at 1 instead of compressing the background.

    This is the normalization used to train the ensemble; it is applied twice,
    once to the full difference image and once to each cutout (see
    :func:`make_model_cutout`).

    Parameters
    ----------
    data : numpy.ndarray
        Image of any shape. May contain NaN or inf.

    Returns
    -------
    numpy.ndarray
        Same shape as ``data``, float32, values in [0, 1]. NaN/inf pixels are
        set to 0. If vmax <= vmin (e.g. a constant image) every pixel is 0.
        If no pixel is finite, an all-zero array with ``data``'s dtype is
        returned.
    """
    from astropy.visualization import ZScaleInterval

    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return np.zeros_like(data)
    # Empty / all-NaN input is handled above, and ZScaleInterval.get_limits
    # does not raise on non-empty finite data, so no exception handling here.
    vmin, vmax = ZScaleInterval().get_limits(data[valid_mask])
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    normalized[~valid_mask] = 0
    return normalized.astype(np.float32)


def create_cutout(data, center_y, center_x, cutout_size=64):
    """Extract a square cutout centred on a (possibly sub-pixel) position.

    Cutout pixel (i, j) samples the image at
    ``(center_y - cutout_size // 2 + i, center_x - cutout_size // 2 + j)``,
    so for even ``cutout_size`` the centre falls on pixel
    ``(cutout_size // 2, cutout_size // 2)``. Values between pixel centres are
    bilinearly interpolated with ``scipy.interpolate.RegularGridInterpolator``;
    for integer centres this reduces to a plain slice of the image.

    Parameters
    ----------
    data : numpy.ndarray
        2D image, indexed ``data[y, x]``.
    center_y, center_x : float
        Cutout centre in 0-indexed pixel coordinates.
    cutout_size : int, optional
        Side length of the cutout in pixels. Default 64, the size the ensemble
        was trained on.

    Returns
    -------
    numpy.ndarray
        Shape (cutout_size, cutout_size), same dtype as ``data``. Positions
        outside the image (beyond the outermost pixel centres) are 0, so
        cutouts near the edge are zero-padded.

    Raises
    ------
    ValueError
        If no sample position of the cutout falls on the image, i.e. the
        cutout would be entirely padding. The check uses the same sample grid
        as the interpolation, so a cutout is rejected exactly when it would
        contain no image pixels.
    """
    half_size = cutout_size // 2
    y_coords = np.arange(cutout_size) + center_y - half_size
    x_coords = np.arange(cutout_size) + center_x - half_size

    ny, nx = data.shape
    if (y_coords[-1] < 0 or y_coords[0] > ny - 1 or
            x_coords[-1] < 0 or x_coords[0] > nx - 1):
        raise ValueError(f"cutout centred at (y={center_y}, x={center_x}) does not overlap "
                         f"the {ny}x{nx} image")

    interpolator = RegularGridInterpolator(
        (np.arange(data.shape[0]), np.arange(data.shape[1])),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    return interpolator((yy, xx)).astype(data.dtype)


def load_fits_2d_ensemble(fits_path, nan_to_zero=True):
    """Read the 2D image from a FITS file as float64, with NaN/inf set to 0.

    Unlike reading ``hdul[0].data`` directly, this also handles sidecar's
    multi-extension products: ``decorr_diff_*.fits`` has an empty primary HDU
    and keeps the image in a ``DATA`` extension (followed by ``NOISE``,
    ``FLAGS`` and ``MASK``).

    The image is taken from, in order of preference:

    1. the ``DATA`` extension, if present;
    2. otherwise the ``SCI`` extension, if present;
    3. otherwise the first HDU whose data is not None.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file.
    nan_to_zero : bool, optional
        If True (default), NaN and +/-inf pixels are replaced by 0. If False
        they are kept, so that :func:`normalize_with_zscale` leaves them out
        of the z-scale limits (as for the science and template channels of the
        3-channel ensemble).

    Returns
    -------
    numpy.ndarray
        2D float64 image. A 3D cube is reduced to its first plane; arrays with
        more dimensions are squeezed. NaN and +/-inf pixels are replaced by 0
        unless ``nan_to_zero`` is False.

    Raises
    ------
    ValueError
        If no HDU in the file contains data.
    OSError
        If the file cannot be opened or is not valid FITS (raised by astropy).
    """
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
    if not nan_to_zero:
        return data
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def to_model_input(cutout_2d):
    """Turn a single-channel cutout into the 3-channel tensor the models expect.

    The models are standard image architectures with a 3-channel (RGB) input
    layer; the ensemble was trained with the same cutout repeated in all three
    channels.

    Parameters
    ----------
    cutout_2d : numpy.ndarray
        2D cutout of shape (H, W), already normalized to [0, 1].

    Returns
    -------
    torch.Tensor
        float32 tensor of shape (3, H, W). Add a batch dimension with
        ``.unsqueeze(0)`` before passing it to a model.
    """
    return torch.from_numpy(cutout_2d).float().unsqueeze(0).repeat(3, 1, 1)


class TimmClassifier(nn.Module):
    """A timm CNN backbone with the ensemble's small classification head.

    Used for the ResNeXt50, RegNetY016, EfficientNetB0 and ConvNeXtTiny
    families (timm names in ``TIMM_NAMES``). The backbone is created without
    its own classifier and with global average pooling, then followed by
    Linear(num_features, 256) - ReLU - Dropout - Linear(256, num_classes) and a
    sigmoid. The layer names match the Hugging Face checkpoints.

    Parameters
    ----------
    model_name : str
        timm model name, e.g. ``"resnext50_32x4d"``.
    num_classes : int, optional
        Number of outputs. The ensemble uses 1 (probability of "real").
    dropout : float, optional
        Dropout probability in the head (only active in training mode).

    Notes
    -----
    ``pretrained=False``: no ImageNet weights are downloaded; the trained
    weights are loaded afterwards from the ensemble checkpoint.
    """

    def __init__(self, model_name, num_classes=1, dropout=0.3):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=False,
                                           num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        """Probability that each input cutout is a real transient.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3, H, W), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (batch,) for ``num_classes=1``, values in [0, 1].
        """
        return torch.sigmoid(self.classifier(self.backbone(x))).squeeze(1)


class DeiTClassifier(nn.Module):
    """DeiT-Tiny vision transformer with the ensemble's classification head.

    Used for the DeiTTiny family. The transformer is built for 64x64 inputs
    (``img_size=64``), i.e. a 4x4 grid of 16x16-pixel patches, so it only
    accepts 64x64 cutouts. The head is the same as in
    :class:`TimmClassifier` with dropout 0.3.

    Parameters
    ----------
    num_classes : int, optional
        Number of outputs. The ensemble uses 1 (probability of "real").

    Notes
    -----
    ``pretrained=False``: the trained weights are loaded afterwards from the
    ensemble checkpoint.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        import timm
        self.deit = timm.create_model("deit_tiny_patch16_224", pretrained=False,
                                       num_classes=0, global_pool="avg", img_size=64)
        nf = self.deit.num_features
        self.classifier = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        """Probability that each input cutout is a real transient.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3, 64, 64), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (batch,) for ``num_classes=1``, values in [0, 1].
        """
        return torch.sigmoid(self.classifier(self.deit(x))).squeeze(1)


def build_ensemble_model(family):
    """Create an untrained model with the architecture of one ensemble family.

    Parameters
    ----------
    family : str
        One of ``ALL_FAMILIES``: ``"DenseNet169"``, ``"ResNeXt50"``,
        ``"RegNetY016"``, ``"EfficientNetB0"``, ``"ConvNeXtTiny"`` or
        ``"DeiTTiny"``.

    Returns
    -------
    torch.nn.Module
        Randomly initialised model in training mode, with one output.

    Raises
    ------
    KeyError
        If ``family`` is not a known family name.
    """
    if family == "DenseNet169":
        return densenet169(num_classes=1)  # reuses this file's existing DenseNet
    if family == "DeiTTiny":
        return DeiTClassifier()
    return TimmClassifier(TIMM_NAMES[family])


def family_is_cached(model_dir, family):
    """Check whether all 4 checkpoints of a family are already on disk.

    Parameters
    ----------
    model_dir : str or Path
        Local model cache directory.
    family : str
        Family name, e.g. ``"DenseNet169"``.

    Returns
    -------
    bool
        True if exactly 4 files match
        ``<model_dir>/<family>/<family>_Ensemble_Model*_best.pth``.
    """
    return len(glob.glob(os.path.join(model_dir, family, f"{family}_Ensemble_Model*_best.pth"))) == 4


def ensure_models_downloaded(model_dir, families):
    """Download any missing ensemble families from Hugging Face.

    Families that are already complete in ``model_dir`` (see
    :func:`family_is_cached`) are not downloaded again. For the missing ones,
    their ``*.pth`` checkpoints and README files (plus the repository README)
    are fetched from ``REPO_ID`` into ``model_dir`` with
    ``huggingface_hub.snapshot_download``.

    Parameters
    ----------
    model_dir : str or Path
        Local cache directory. Created if it does not exist.
    families : list of str
        Families that are needed.

    Returns
    -------
    str or Path
        Directory containing the models: ``model_dir`` itself if nothing had
        to be downloaded, otherwise the path returned by ``snapshot_download``
        (the same location).

    Notes
    -----
    Needs network access and Hugging Face credentials that can read
    ``REPO_ID`` (e.g. via ``hf auth login`` or the ``HF_TOKEN`` environment
    variable), since the repository is not publicly readable.

    This is meant to be run once per system, before launching pipeline jobs
    (``python -m sidecar.classifier --download-models``), not from inside
    the jobs: many jobs downloading into the same ``model_dir`` at once would
    race each other, and jobs on fresh containers would each download their
    own copy.
    """
    missing = [fam for fam in families if not family_is_cached(model_dir, fam)]
    if not missing:
        return model_dir
    # Imported here so that pipeline jobs, which only read the models, do not
    # need huggingface_hub.
    from huggingface_hub import snapshot_download
    patterns = ["README.md"] + [p for fam in missing for p in (f"{fam}/*.pth", f"{fam}/README.md")]
    return snapshot_download(repo_id=REPO_ID, local_dir=model_dir, allow_patterns=patterns)


def check_models_available(model_dir, families):
    """Raise if any requested family is missing from ``model_dir``.

    Parameters
    ----------
    model_dir : str or Path
        Local model directory.
    families : list of str
        Families that are needed.

    Raises
    ------
    FileNotFoundError
        If any family in ``families`` does not have all its checkpoints in
        ``model_dir`` (see :func:`family_is_cached`). The message says how to
        download them.
    """
    missing = [fam for fam in families if not family_is_cached(model_dir, fam)]
    if missing:
        raise FileNotFoundError(
            f"Real/bogus models missing from {model_dir}: {', '.join(missing)}. "
            f"Download them once before running the pipeline with "
            f"'python -m sidecar.classifier --download-models --model-dir {model_dir}', "
            f"or run the pipeline with --no-real-bogus."
        )


def default_model_dir():
    """Return the configured model directory.

    Returns
    -------
    str
        ``photometry.sidecar.real_bogus.model_dir`` from the default snappl
        config (by default ``${system.paths.dev_storage}/sidecar/real_bogus_models``).
    """
    return Config.get().value("photometry.sidecar.real_bogus.model_dir")


def load_ensemble(model_dir, families, device, download=False):
    """Load the trained models of the requested families.

    For each family, every checkpoint matching
    ``<model_dir>/<family>/<family>_Ensemble_Model*_best.pth`` is loaded (in
    sorted filename order) into a fresh model from
    :func:`build_ensemble_model`, moved to ``device`` and put in eval mode.
    Each checkpoint is a dict whose ``"model_state_dict"`` holds the weights.

    Parameters
    ----------
    model_dir : str or Path
        Local model directory (see :func:`ensure_models_downloaded`).
    families : list of str
        Families to load, from ``ALL_FAMILIES``.
    device : torch.device
        Device to place the models on.
    download : bool, optional
        If True, download missing families first with
        :func:`ensure_models_downloaded`. Default False: missing families
        raise instead (see :func:`check_models_available`), so that parallel
        pipeline jobs never download.

    Returns
    -------
    dict of str to list of torch.nn.Module
        ``{family: [model, ...]}`` in the order of ``families``.

    Raises
    ------
    FileNotFoundError
        If ``download`` is False and a requested family is not in
        ``model_dir``.
    RuntimeError
        If no model at all could be loaded.
    """
    if download:
        model_dir = ensure_models_downloaded(model_dir, families)
    else:
        check_models_available(model_dir, families)
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
    """Score one cutout with every model and combine the results.

    The combination is an "ensemble of ensembles": the members of each family
    are averaged into one per-family score, and the overall score is the
    unweighted mean of the per-family scores, so every architecture counts
    equally regardless of how many members it has.

    Parameters
    ----------
    family_models : dict of str to list of torch.nn.Module
        Output of :func:`load_ensemble`.
    cutout_2d_normalized : numpy.ndarray
        2D cutout prepared with :func:`make_model_cutout`.
    device : torch.device
        Device the models are on.

    Returns
    -------
    per_family_mean : dict of str to float
        Mean probability of "real" for each family.
    overall_mean : float
        Mean of the per-family scores, in [0, 1].
    overall_std : float
        Standard deviation across all individual model outputs (not across
        the family means): a measure of how much the models disagree.
    """
    x = to_model_input(cutout_2d_normalized).unsqueeze(0).to(device)
    return _ensemble_scores(family_models, x)


def _ensemble_scores(family_models, x):
    """Score one model-ready input tensor of shape (1, 3, H, W); see :func:`predict_ensemble`."""
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


# ============================================================================
# PER-DETECTION SCORING
# ============================================================================


def resolve_device(device="auto"):
    """Turn a device choice into the ``torch.device`` the models will run on.

    Lets the caller force CPU inference even on a machine where a GPU is
    available (e.g. to leave the GPU free, or to compare results).

    Parameters
    ----------
    device : {"auto", "cpu", "cuda"} or None, optional
        ``"auto"`` (or None): use the GPU if PyTorch can see one, otherwise the
        CPU. ``"cpu"``: always use the CPU. ``"cuda"``: always use the GPU.
        Case-insensitive. Default ``"auto"``.

    Returns
    -------
    torch.device

    Raises
    ------
    ValueError
        If ``device`` is not one of the accepted values.
    RuntimeError
        If ``"cuda"`` is requested but PyTorch cannot see a GPU (for example a
        CPU-only PyTorch build), rather than silently falling back to the CPU.
    """
    choice = "auto" if device is None else str(device).lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but PyTorch cannot see a GPU")
        return torch.device("cuda")
    raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', not {device!r}")


def get_cutout_center(row, x_col, y_col):
    """Read one catalog row's position as a 0-indexed (y, x) pixel position.

    Only reads and converts the coordinates; whether the position is usable
    for a cutout is decided by :func:`create_cutout`, which builds the cutout
    grid.

    ``X_IMAGE``/``Y_IMAGE`` (Source Extractor) are 1-indexed and are shifted
    by -1; any other columns (e.g. ``x_centroid``/``y_centroid``) are taken as
    already 0-indexed.

    Parameters
    ----------
    row : astropy.table.Row
        One row of the detection catalog.
    x_col, y_col : str
        Names of the position columns, from :func:`detect_catalog_type`.

    Returns
    -------
    y_center, x_center : float
        Position in 0-indexed pixel coordinates.

    Raises
    ------
    ValueError
        If a coordinate is non-numeric, masked, NaN or infinite.
    """
    try:
        x_image = float(row[x_col])
        y_image = float(row[y_col])
    except TypeError as e:
        raise ValueError(f"non-numeric coordinates: {e}") from e
    if not (np.isfinite(x_image) and np.isfinite(y_image)):
        raise ValueError("non-finite coordinates")

    if x_col in ('X_IMAGE', 'Y_IMAGE'):
        return y_image - 1.0, x_image - 1.0
    return y_image, x_image


def make_model_cutout(row, x_col, y_col, normalized_image, cutout_size=64):
    """Make the model-ready cutout for one catalog row.

    This is the single place that turns a catalog row into the input the
    ensemble sees, so every detection is prepared the same way:

    1. :func:`get_cutout_center` reads the row's position and converts it to
       0-indexed pixels;
    2. :func:`create_cutout` takes a sub-pixel bilinear cutout centred there
       from an image that has already been z-scaled once, rejecting positions
       whose cutout would not overlap the image;
    3. :func:`normalize_with_zscale` z-scales the cutout again, so its
       contrast is set by its own pixels (the double z-scale used in
       training).

    Parameters
    ----------
    row : astropy.table.Row
        One row of the detection catalog.
    x_col, y_col : str
        Names of the position columns, from :func:`detect_catalog_type`.
    normalized_image : numpy.ndarray
        Full 2D difference image, already passed through
        :func:`normalize_with_zscale`.
    cutout_size : int, optional
        Cutout side length in pixels. Default 64; must be 64 for the
        published ensemble.

    Returns
    -------
    numpy.ndarray
        float32 array of shape (cutout_size, cutout_size), values in [0, 1].

    Raises
    ------
    ValueError
        If the row's coordinates are unusable (non-numeric, masked, NaN/inf)
        or the cutout would contain no image pixels.
    """
    y_center, x_center = get_cutout_center(row, x_col, y_col)
    cutout = create_cutout(normalized_image, y_center, x_center, cutout_size)
    return normalize_with_zscale(cutout.astype(np.float32))


def score_detection(row, x_col, y_col, normalized_image, family_models, device,
                    threshold=0.5, cutout_size=64):
    """Compute the real/bogus scores for one catalog row.

    Makes the model-ready cutout (:func:`make_model_cutout`) and scores it
    (:func:`predict_ensemble`). Rows whose coordinates are unusable, or whose
    cutout would contain no image pixels, are not scored and are labelled
    "invalid" instead of raising, so one bad row does not stop a catalog.
    Errors from the models are not caught.

    Parameters
    ----------
    row : astropy.table.Row
        One row of the detection catalog.
    x_col, y_col : str
        Names of the position columns, from :func:`detect_catalog_type`.
    normalized_image : numpy.ndarray
        Full 2D difference image, already passed through
        :func:`normalize_with_zscale`.
    family_models : dict of str to list of torch.nn.Module
        Output of :func:`load_ensemble`.
    device : torch.device
        Device the models are on.
    threshold : float, optional
        A detection is labelled "real" if its ``real_bogus_score`` is
        >= threshold, otherwise "bogus". Default 0.5.
    cutout_size : int, optional
        Cutout side length in pixels. Default 64.

    Returns
    -------
    dict
        One entry per output catalog column, in this order:

        * ``rb_<family>`` (float) for each family in ``family_models``: mean
          probability of "real" over that family's members;
        * ``real_bogus_score`` (float): mean of the per-family scores;
        * ``real_bogus_score_std`` (float): spread across all individual
          models;
        * ``prediction`` (str): "real", "bogus", or "invalid" if no cutout
          could be made for the row, in which case all scores are NaN.
    """
    try:
        cutout = make_model_cutout(row, x_col, y_col, normalized_image, cutout_size)
    except ValueError:
        result = {f"rb_{fam}": np.nan for fam in family_models}
        result.update(real_bogus_score=np.nan, real_bogus_score_std=np.nan, prediction="invalid")
        return result

    per_family_mean, overall_mean, overall_std = predict_ensemble(family_models, cutout, device)

    result = {f"rb_{fam}": per_family_mean[fam] for fam in family_models}
    result.update(
        real_bogus_score=overall_mean,
        real_bogus_score_std=overall_std,
        prediction="real" if overall_mean >= threshold else "bogus",
    )
    return result


def process_single_catalog_ensemble(catalog_path, fits_path, output_path,
                                     model_dir=None, families=None,
                                     cutout_size=64, threshold=0.5, device="auto",
                                     output_format="parquet"):
    """Add real/bogus scores for every detection in a catalog and save it.

    Loads the ensemble from ``model_dir`` (:func:`load_ensemble`; nothing is
    downloaded, see :func:`main`), reads the catalog and the difference image, z-scales the
    whole image once, then scores each row with :func:`score_detection`.
    Cutouts are made in memory; no cutout files are written. The input
    catalog is not modified; the result is written to ``output_path``.

    Parameters
    ----------
    catalog_path : str or Path
        Input detection catalog, ECSV or Parquet (read with
        :func:`sidecar.catalog_io.read_catalog`, by file suffix). Must have ``x_centroid``/``y_centroid``
        or ``X_IMAGE``/``Y_IMAGE`` columns (see :func:`detect_catalog_type`).
    fits_path : str or Path
        Difference image the detections were measured on, e.g. sidecar's
        ``decorr_diff_*.fits`` (read with :func:`load_fits_2d_ensemble`).
        Catalog positions must be in this image's pixel coordinates.
    output_path : str or Path
        Where to write the scored catalog. Overwritten if it exists.
        Table metadata (threshold, model dir, etc.) is kept in the file.
    model_dir : str or Path, optional
        Directory holding the ensemble weights. Default: the configured
        ``photometry.sidecar.real_bogus.model_dir`` (see :func:`default_model_dir`).
    families : list of str, optional
        Subset of ``ALL_FAMILIES`` to use. Default: all 6.
    cutout_size : int, optional
        Cutout side length in pixels. Default 64; must be 64 for the published
        ensemble (the DeiT model only accepts 64x64 input).
    threshold : float, optional
        A detection is labelled "real" if its ``real_bogus_score`` is
        >= threshold, otherwise "bogus". Default 0.5.
    device : {"auto", "cpu", "cuda"}, optional
        Where to run the models (see :func:`resolve_device`). Default
        ``"auto"``: the GPU if PyTorch can see one, otherwise the CPU. Use
        ``"cpu"`` to force CPU inference on a GPU machine.
    output_format : {"parquet", "ecsv"}, optional
        Format of the output catalog (see
        :func:`sidecar.catalog_io.write_catalog`). Default ``"parquet"``.

    Returns
    -------
    None
        The output catalog has all input columns plus, per row (see
        :func:`score_detection`): ``rb_<family>`` for each loaded family,
        ``real_bogus_score``, ``real_bogus_score_std`` and ``prediction``.
        Its metadata records ``real_bogus_model`` ("diff"), ``real_bogus_threshold``, ``real_bogus_model_dir``,
        ``real_bogus_n_models`` and ``real_bogus_cutout_size``.

    Raises
    ------
    ValueError
        If the catalog has no recognised position columns, the FITS file has
        no image data, ``device`` is not "auto", "cpu" or "cuda", or
        ``output_format`` is not one of ``OUTPUT_FORMATS``.
    FileNotFoundError
        If a requested model family is not in ``model_dir``.
    RuntimeError
        If no models could be loaded, or ``device="cuda"`` but no GPU is
        available.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, not {output_format!r}")
    if model_dir is None:
        model_dir = default_model_dir()
    if families is None:
        families = ALL_FAMILIES

    device = resolve_device(device)
    family_models = load_ensemble(model_dir, families, device)
    total_models = sum(len(v) for v in family_models.values())

    data = read_catalog(catalog_path)
    catalog_type, x_col, y_col, id_col = detect_catalog_type(data)

    normalized_full = normalize_with_zscale(load_fits_2d_ensemble(fits_path))

    n_rows = len(data)
    log_every = max(1, n_rows // 10)
    SNLogger.info(f"Scoring {n_rows} detections with {total_models} models")
    results = []
    for i, row in enumerate(data, start=1):
        results.append(score_detection(row, x_col, y_col, normalized_full, family_models, device,
                                       threshold=threshold, cutout_size=cutout_size))
        if i % log_every == 0 or i == n_rows:
            SNLogger.info(f"Scored {i}/{n_rows} detections")
    output_columns = list(results[0]) if results else [f"rb_{fam}" for fam in family_models] + [
        "real_bogus_score", "real_bogus_score_std", "prediction"]
    for col in output_columns:
        data[col] = [r[col] for r in results]
    data.meta["real_bogus_model"] = "diff"
    data.meta["real_bogus_threshold"] = threshold
    data.meta["real_bogus_model_dir"] = str(model_dir)
    data.meta["real_bogus_n_models"] = total_models
    data.meta["real_bogus_cutout_size"] = cutout_size

    write_catalog(data, output_path, overwrite=True, output_format=output_format)


# ============================================================================
# 3-CHANNEL ENSEMBLE (Roman-Supernova-PIT/Real-Bogus-3ch on Hugging Face)
# 2 architecture families x 4 members = 8 models. Input channels, in this
# order: science, template, diff, all on the same pixel grid. Each channel is
# prepared like the diff of the 1-channel ensemble (z-scale the full image,
# 64x64 bilinear cutout, z-scale the cutout again). The models were trained
# image-only: their "meta" input is always 0.
# ============================================================================

REPO_ID_3CH = "Roman-Supernova-PIT/Real-Bogus-3ch"
FAMILIES_3CH = {"DenseNet121": "densenet121", "ConvNeXtTiny": "convnext_tiny"}
CHANNELS_3CH = ("science", "template", "diff")
# Model version = top-level folder in REPO_ID_3CH (and in the local model_dir_3ch), e.g.
# nov_24_2026/DenseNet121_3ch_meta/... .  photometry.sidecar.real_bogus.version_3ch overrides it.
VERSION_3CH = "nov_24_2026"


class ThreeChannelClassifier(nn.Module):
    """A timm backbone plus the (unused) meta branch of the 3-channel ensemble.

    The layer names and sizes match the ``Real-Bogus-3ch`` checkpoints:
    backbone features are concatenated with the output of a small MLP on the
    "meta" input, followed by Linear(num_features + 64, 256) - ReLU - Dropout -
    Linear(256, 1). The published models were trained without meta features,
    so the meta input is always a vector of zeros.

    Parameters
    ----------
    model_name : str
        timm model name, e.g. ``"convnext_tiny"``.
    n_meta : int, optional
        Length of the meta input stored in the checkpoint. Default 1.
    dropout : float, optional
        Dropout probability in the head (only active in training mode).
    """

    def __init__(self, model_name, n_meta=1, dropout=0.3):
        super().__init__()
        import timm
        self.n_meta = n_meta
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        nf = self.backbone.num_features
        self.meta = nn.Sequential(nn.Linear(n_meta, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(nf + 64, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        """Probability that each input stamp is a real transient.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3, 64, 64): science, template, diff, values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (batch,), values in [0, 1].
        """
        meta = x.new_zeros((x.shape[0], self.n_meta))
        features = torch.cat([self.backbone(x), self.meta(meta)], 1)
        return torch.sigmoid(self.classifier(features)).squeeze(1)


def checkpoints_3ch(model_dir, family, version=VERSION_3CH):
    """List the checkpoints of one 3-channel family in ``model_dir``.

    Parameters
    ----------
    model_dir : str or Path
        Local model directory.
    family : str
        One of ``FAMILIES_3CH``.
    version : str, optional
        Model version (sub-folder of ``model_dir``). Default ``VERSION_3CH``.

    Returns
    -------
    list of str
        Sorted paths matching
        ``<model_dir>/<version>/<family>_3ch_meta/<family>_3ch_meta_Ensemble_Model*_best*.pth``.
    """
    pattern = os.path.join(model_dir, version, f"{family}_3ch_meta",
                           f"{family}_3ch_meta_Ensemble_Model*_best*.pth")
    return sorted(glob.glob(pattern))


def check_models_available_3ch(model_dir, version=VERSION_3CH):
    """Raise if any 3-channel family does not have exactly 4 checkpoints in ``model_dir``.

    Parameters
    ----------
    model_dir : str or Path
        Local model directory.
    version : str, optional
        Model version (sub-folder of ``model_dir``). Default ``VERSION_3CH``.

    Raises
    ------
    FileNotFoundError
        If a family is missing or incomplete. The message says how to download it.
    """
    missing = [fam for fam in FAMILIES_3CH if len(checkpoints_3ch(model_dir, fam, version)) != 4]
    if missing:
        raise FileNotFoundError(
            f"3-channel real/bogus models (version {version}) missing from {model_dir}: {', '.join(missing)}. "
            f"Download them once before running the pipeline with "
            f"'python -m sidecar.classifier --download-models --model-type 3ch --model-dir {model_dir}', "
            f"or run the pipeline with --no-real-bogus."
        )


def ensure_models_downloaded_3ch(model_dir, version=VERSION_3CH):
    """Download the 3-channel ensemble from ``REPO_ID_3CH`` if it is not complete in ``model_dir``.

    Like :func:`ensure_models_downloaded`, this is meant to be run once per
    system (``python -m sidecar.classifier --download-models --model-type 3ch``),
    never from pipeline jobs. Needs Hugging Face credentials that can read
    ``REPO_ID_3CH``.

    Parameters
    ----------
    model_dir : str or Path
        Local directory. Created if it does not exist. The repository layout
        is kept, so the models land in ``<model_dir>/<version>/``.
    version : str, optional
        Model version (top-level folder of ``REPO_ID_3CH``). Default ``VERSION_3CH``.
    """
    missing = [fam for fam in FAMILIES_3CH if len(checkpoints_3ch(model_dir, fam, version)) != 4]
    if not missing:
        return
    from huggingface_hub import snapshot_download
    patterns = ["README.md"] + [f"{version}/{fam}_3ch_meta/*.pth" for fam in missing]
    snapshot_download(repo_id=REPO_ID_3CH, local_dir=model_dir, allow_patterns=patterns)


def default_model_dir_3ch():
    """Return ``photometry.sidecar.real_bogus.model_dir_3ch`` from the default snappl config."""
    return Config.get().value("photometry.sidecar.real_bogus.model_dir_3ch")


def default_version_3ch():
    """Return ``photometry.sidecar.real_bogus.version_3ch`` from the default snappl config, else ``VERSION_3CH``."""
    return Config.get().value("photometry.sidecar.real_bogus.version_3ch", default=VERSION_3CH)


def load_ensemble_3ch(model_dir, device, version=VERSION_3CH):
    """Load the 8 models of the 3-channel ensemble.

    Parameters
    ----------
    model_dir : str or Path
        Local model directory (see :func:`ensure_models_downloaded_3ch`).
    device : torch.device
        Device to place the models on.
    version : str, optional
        Model version (sub-folder of ``model_dir``). Default ``VERSION_3CH``.

    Returns
    -------
    dict of str to list of torch.nn.Module
        ``{family: [model, ...]}``, models in eval mode.

    Raises
    ------
    FileNotFoundError
        If a family is not complete in ``model_dir``.
    ValueError
        If a checkpoint was trained with meta features, which this code does
        not provide.
    """
    check_models_available_3ch(model_dir, version)
    family_models = {}
    for fam, timm_name in FAMILIES_3CH.items():
        models = []
        for path in checkpoints_3ch(model_dir, fam, version):
            ckpt = torch.load(path, map_location=device)
            if not ckpt.get("features", {}).get("image_only", False):
                raise ValueError(f"{path} was trained with meta features; only image-only models are supported")
            model = ThreeChannelClassifier(timm_name, n_meta=ckpt["n_meta"]).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
        family_models[fam] = models
    return family_models


def make_model_cutout_3ch(row, x_col, y_col, normalized_images, cutout_size=64):
    """Make the (science, template, diff) stamp for one catalog row.

    Each channel is made with :func:`make_model_cutout` at the same position,
    so all three get the same sub-pixel cutout and second z-scale.

    Parameters
    ----------
    row : astropy.table.Row
        One row of the detection catalog.
    x_col, y_col : str
        Names of the position columns, from :func:`detect_catalog_type`.
    normalized_images : sequence of 3 numpy.ndarray
        Full science, template and diff images on the same pixel grid, each
        already passed through :func:`normalize_with_zscale`.
    cutout_size : int, optional
        Cutout side length in pixels. Default 64.

    Returns
    -------
    numpy.ndarray
        float32 array of shape (3, cutout_size, cutout_size), values in [0, 1].

    Raises
    ------
    ValueError
        If the row's coordinates are unusable or the cutout would contain no
        image pixels.
    """
    return np.stack([make_model_cutout(row, x_col, y_col, img, cutout_size) for img in normalized_images])


def score_detection_3ch(row, x_col, y_col, normalized_images, family_models, device,
                        threshold=0.5, cutout_size=64):
    """Compute the 3-channel real/bogus scores for one catalog row.

    Same output as :func:`score_detection` (``rb_<family>``,
    ``real_bogus_score``, ``real_bogus_score_std``, ``prediction``), with the
    stamp made by :func:`make_model_cutout_3ch`. Rows without a usable cutout
    are labelled "invalid" with NaN scores.

    Parameters
    ----------
    row, x_col, y_col, threshold, cutout_size
        As for :func:`score_detection`.
    normalized_images : sequence of 3 numpy.ndarray
        See :func:`make_model_cutout_3ch`.
    family_models : dict of str to list of torch.nn.Module
        Output of :func:`load_ensemble_3ch`.
    device : torch.device
        Device the models are on.

    Returns
    -------
    dict
        One entry per output catalog column.
    """
    try:
        stamp = make_model_cutout_3ch(row, x_col, y_col, normalized_images, cutout_size)
    except ValueError:
        result = {f"rb_{fam}": np.nan for fam in family_models}
        result.update(real_bogus_score=np.nan, real_bogus_score_std=np.nan, prediction="invalid")
        return result

    x = torch.from_numpy(stamp).unsqueeze(0).to(device)
    per_family_mean, overall_mean, overall_std = _ensemble_scores(family_models, x)
    result = {f"rb_{fam}": per_family_mean[fam] for fam in family_models}
    result.update(
        real_bogus_score=overall_mean,
        real_bogus_score_std=overall_std,
        prediction="real" if overall_mean >= threshold else "bogus",
    )
    return result


def process_single_catalog_ensemble_3ch(catalog_path, science_path, template_path, diff_path, output_path,
                                         model_dir=None, cutout_size=64, threshold=0.5, device="auto",
                                         output_format="parquet", version=None):
    """Add 3-channel real/bogus scores for every detection in a catalog and save it.

    The 3-channel counterpart of :func:`process_single_catalog_ensemble`: same
    output columns and metadata, but each detection is scored on a
    (science, template, diff) stamp. The three images must be on the same
    pixel grid as the catalog positions (sidecar writes the sky-subtracted
    science and the template resampled onto the science grid for this, see
    ``subtraction.Pipeline``).

    Each full image is z-scaled once before cutting, as in training. The diff
    has NaN/inf set to 0 first; in the science and template, NaN/inf pixels
    are left out of the z-scale limits and set to 0 afterwards.

    Parameters
    ----------
    catalog_path : str or Path
        Input detection catalog, ECSV or Parquet.
    science_path, template_path, diff_path : str or Path
        Science, resampled template and difference images (FITS).
    output_path : str or Path
        Where to write the scored catalog. Overwritten if it exists.
    model_dir : str or Path, optional
        Directory holding the 3-channel weights. Default: the configured
        ``photometry.sidecar.real_bogus.model_dir_3ch``.
    version : str, optional
        Model version, the sub-folder of ``model_dir`` holding the families.
        Default: the configured ``photometry.sidecar.real_bogus.version_3ch``
        (``VERSION_3CH`` if not set).
    cutout_size : int, optional
        Cutout side length in pixels. Default 64; must be 64 for the published models.
    threshold : float, optional
        A detection is labelled "real" if its ``real_bogus_score`` is >= threshold. Default 0.5.
    device : {"auto", "cpu", "cuda"}, optional
        Where to run the models (see :func:`resolve_device`).
    output_format : {"parquet", "ecsv"}, optional
        Format of the output catalog. Default ``"parquet"``.

    Raises
    ------
    ValueError
        If ``output_format`` is unknown, the catalog has no position columns,
        the images do not all have the same shape, or a checkpoint needs meta features.
    FileNotFoundError
        If the models are not in ``model_dir``.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, not {output_format!r}")
    if model_dir is None:
        model_dir = default_model_dir_3ch()
    if version is None:
        version = default_version_3ch()

    device = resolve_device(device)
    family_models = load_ensemble_3ch(model_dir, device, version)
    total_models = sum(len(v) for v in family_models.values())

    data = read_catalog(catalog_path)
    catalog_type, x_col, y_col, id_col = detect_catalog_type(data)

    normalized_images = [
        normalize_with_zscale(load_fits_2d_ensemble(science_path, nan_to_zero=False)),
        normalize_with_zscale(load_fits_2d_ensemble(template_path, nan_to_zero=False)),
        normalize_with_zscale(load_fits_2d_ensemble(diff_path)),
    ]
    if len({img.shape for img in normalized_images}) != 1:
        raise ValueError(f"science, template and diff images differ in shape: "
                         f"{[img.shape for img in normalized_images]}")

    n_rows = len(data)
    log_every = max(1, n_rows // 10)
    SNLogger.info(f"Scoring {n_rows} detections with {total_models} 3-channel models")
    results = []
    for i, row in enumerate(data, start=1):
        results.append(score_detection_3ch(row, x_col, y_col, normalized_images, family_models, device,
                                           threshold=threshold, cutout_size=cutout_size))
        if i % log_every == 0 or i == n_rows:
            SNLogger.info(f"Scored {i}/{n_rows} detections")
    output_columns = list(results[0]) if results else [f"rb_{fam}" for fam in family_models] + [
        "real_bogus_score", "real_bogus_score_std", "prediction"]
    for col in output_columns:
        data[col] = [r[col] for r in results]
    data.meta["real_bogus_model"] = "3ch"
    data.meta["real_bogus_version"] = version
    data.meta["real_bogus_threshold"] = threshold
    data.meta["real_bogus_model_dir"] = str(model_dir)
    data.meta["real_bogus_n_models"] = total_models
    data.meta["real_bogus_cutout_size"] = cutout_size

    write_catalog(data, output_path, overwrite=True, output_format=output_format)


def main(argv=None):
    """Command-line entry point: download the model ensemble once.

    Run this once per system, before launching pipeline jobs, so that the
    jobs can all read the same models from ``model_dir`` without any of them
    downloading::

        python -m sidecar.classifier --download-models
        python -m sidecar.classifier --download-models --model-type 3ch   # 3-channel ensemble

    Needs network access and Hugging Face credentials for ``REPO_ID``
    (or ``REPO_ID_3CH``) (``hf auth login`` or ``HF_TOKEN``). Families already
    present are skipped.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse instead of ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description="Manage the real/bogus CNN model ensemble used by the sidecar pipeline."
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help=f"Download the model ensemble from Hugging Face ({REPO_ID}) into --model-dir.",
    )
    parser.add_argument(
        "--model-type",
        default="diff",
        choices=["diff", "3ch"],
        help=f"'diff': the difference-image ensemble ({REPO_ID}); "
        f"'3ch': the science/template/diff ensemble ({REPO_ID_3CH}).  Default: diff.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Where to put the models.  Defaults to photometry.sidecar.real_bogus.model_dir "
        "(model_dir_3ch with --model-type 3ch) in the config.",
    )
    parser.add_argument(
        "--version-3ch",
        default=None,
        help="Model version to download with --model-type 3ch (top-level folder of the Hugging Face repository).  "
        f"Defaults to photometry.sidecar.real_bogus.version_3ch in the config, else {VERSION_3CH}.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=ALL_FAMILIES,
        choices=ALL_FAMILIES,
        help="Model families to download (--model-type diff only).  Default: all.",
    )
    args = parser.parse_args(argv)
    if not args.download_models:
        parser.error("nothing to do; pass --download-models")

    if args.model_type == "3ch":
        model_dir = args.model_dir if args.model_dir is not None else default_model_dir_3ch()
        version = args.version_3ch if args.version_3ch is not None else default_version_3ch()
        SNLogger.info(f"Downloading 3-channel real/bogus models ({', '.join(FAMILIES_3CH)}, version {version}) "
                      f"into {model_dir}")
        ensure_models_downloaded_3ch(model_dir, version)
        check_models_available_3ch(model_dir, version)
        SNLogger.info(f"3-channel real/bogus models are ready in {model_dir}")
        return

    model_dir = args.model_dir if args.model_dir is not None else default_model_dir()
    SNLogger.info(f"Downloading real/bogus models ({', '.join(args.families)}) into {model_dir}")
    ensure_models_downloaded(model_dir, args.families)
    check_models_available(model_dir, args.families)
    SNLogger.info(f"Real/bogus models are ready in {model_dir}")


if __name__ == "__main__":
    main()
