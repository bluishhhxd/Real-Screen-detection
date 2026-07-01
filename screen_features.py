from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


MAX_SIDE = 2048
PATCH_SIZE = 224
MAX_TILES = 10

try:
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    NEAREST = Image.NEAREST


def sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def load_image(path, max_side=MAX_SIDE):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = image.size
    scale = max(width, height) / float(max_side)
    if scale > 1.0:
        new_size = (
            max(1, int(round(width / scale))),
            max(1, int(round(height / scale))),
        )
        image = image.resize(new_size, NEAREST)
    return np.asarray(image, dtype=np.float32) / 255.0


def gray_chroma(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    c1 = r - g
    c2 = b - 0.5 * (r + g)
    return gray, c1, c2


def blur3(values, reps=1):
    out = values
    for _ in range(reps):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="reflect")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def box1d(values, width):
    if len(values) < width:
        return np.full_like(values, values.mean())

    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="reflect")
    csum = np.cumsum(np.r_[0.0, padded])
    return (csum[width:] - csum[:-width]) / float(width)


def select_tiles(gray, patch_size=PATCH_SIZE, max_tiles=MAX_TILES):
    height, width = gray.shape
    patch = min(patch_size, height, width)
    step = patch
    y0 = int(height * 0.05)
    y1 = int(height * 0.95)
    x0 = int(width * 0.05)
    x1 = int(width * 0.95)
    tiles = []

    for y in range(y0, max(y0 + 1, y1 - patch + 1), step):
        for x in range(x0, max(x0 + 1, x1 - patch + 1), step):
            tile = gray[y : y + patch, x : x + patch]
            if tile.shape != (patch, patch):
                continue

            mean = float(tile.mean())
            if mean < 0.035 or mean > 0.985:
                continue

            grad = float(
                np.mean(np.abs(np.diff(tile, axis=1)))
                + np.mean(np.abs(np.diff(tile, axis=0)))
            )
            std = float(tile.std())
            score = grad + 0.05 * std
            tiles.append((score, y, x, grad, std, mean))

    if not tiles:
        tiles = [
            (
                0.0,
                max(0, (height - patch) // 2),
                max(0, (width - patch) // 2),
                0.0,
                0.0,
                0.0,
            )
        ]

    return sorted(tiles)[: min(max_tiles, len(tiles))], patch


def fft_features(channels, tiles, patch):
    window = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    yy, xx = np.ogrid[:patch, :patch]
    cy = patch // 2
    cx = patch // 2
    rr = np.sqrt(((yy - cy) / patch) ** 2 + ((xx - cx) / patch) ** 2)
    band_mask = (rr > 0.035) & (rr < 0.49)
    high_mask = (rr > 0.16) & (rr < 0.49)
    axis_mask = ((np.abs(yy - cy) < 4) | (np.abs(xx - cx) < 4)) & band_mask
    vertical_mask = (np.abs(yy - cy) < 4) & band_mask
    horizontal_mask = (np.abs(xx - cx) < 4) & band_mask

    features = []
    for channel in channels:
        rows = []
        for _, y, x, *_ in tiles:
            tile = channel[y : y + patch, x : x + patch]
            residual = tile - blur3(tile, 2)
            residual = residual - residual.mean()
            power = np.abs(np.fft.fftshift(np.fft.fft2(residual * window))) ** 2
            band = power[band_mask]
            median = np.median(band) + 1e-12
            abs_residual = np.abs(residual)
            rows.append(
                [
                    np.log1p(np.percentile(band, 99.5) / median),
                    np.log1p(band.max() / median),
                    np.log1p(np.percentile(power[axis_mask], 99.5) / median),
                    np.log1p(np.percentile(power[vertical_mask], 99.5) / median),
                    np.log1p(np.percentile(power[horizontal_mask], 99.5) / median),
                    np.mean(power[high_mask]) / (median + 1e-12),
                    np.percentile(abs_residual, 95),
                    np.mean(abs_residual),
                ]
            )

        values = np.asarray(rows, dtype=np.float64)
        features.extend(values.mean(axis=0))
        features.extend(values.max(axis=0))
        features.extend(values.std(axis=0))

    return [float(value) for value in features]


def global_texture_features(gray, c1, c2):
    features = []
    for channel in (gray, c1, c2):
        residual = channel - blur3(channel, 2)
        dx = np.diff(channel, axis=1)
        dy = np.diff(channel, axis=0)
        ddx = np.diff(channel, n=2, axis=1)
        ddy = np.diff(channel, n=2, axis=0)
        features.extend(
            [
                float(np.mean(np.abs(residual))),
                float(np.percentile(np.abs(residual), 95)),
                float(np.mean(np.abs(dx))),
                float(np.mean(np.abs(dy))),
                float(np.mean(np.abs(ddx))),
                float(np.mean(np.abs(ddy))),
            ]
        )
    return features


def stripe_features(gray, c1, c2):
    features = []
    for channel in (gray, c1, c2, np.abs(c1) + np.abs(c2)):
        for axis in (0, 1):
            signal = channel.mean(axis=axis).astype(np.float64)
            width = max(9, (len(signal) // 80) * 2 + 1)
            signal = signal - box1d(signal, width)
            signal = signal - signal.mean()

            if len(signal) < 32:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
            lo = max(3, len(spectrum) // 80)
            hi = max(lo + 1, int(len(spectrum) * 0.95))
            band = spectrum[lo:hi]
            if len(band) == 0:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            median = np.median(band) + 1e-9
            top = np.sort(band)[-5:]
            features.extend(
                [
                    float(np.log1p(band.max() / median)),
                    float(np.log1p(np.percentile(band, 99) / median)),
                    float(top.sum() / (band.sum() + 1e-9)),
                    float(np.mean(band) / median),
                    float(np.argmax(band) / max(1, len(band))),
                ]
            )

    return features


def border_features(gray):
    height, width = gray.shape
    features = []
    first_edge_means = None

    for frac in (0.04, 0.08, 0.14):
        strip_h = max(2, int(height * frac))
        strip_w = max(2, int(width * frac))
        regions = [
            gray[:strip_h, :],
            gray[-strip_h:, :],
            gray[:, :strip_w],
            gray[:, -strip_w:],
        ]
        means = [float(region.mean()) for region in regions]
        dark = [float((region < 0.08).mean()) for region in regions]
        very_dark = [float((region < 0.035).mean()) for region in regions]
        if first_edge_means is None:
            first_edge_means = means
        features.extend(means)
        features.extend(dark)
        features.extend(very_dark)
        features.extend([max(dark), max(very_dark), min(means), float(np.mean(means))])

    center = gray[height // 5 : 4 * height // 5, width // 5 : 4 * width // 5]
    center_mean = float(center.mean())
    features.extend(
        [
            center_mean,
            float((center < 0.08).mean()),
            float(min(first_edge_means) / (center_mean + 1e-6)),
        ]
    )
    return features


def cursorish_features(gray, rgb):
    white = (gray > 0.92) & (rgb.max(axis=2) - rgb.min(axis=2) < 0.08)
    if white.mean() > 0.2:
        return [0.0, 0.0, 0.0, 0.0]

    highpass = np.abs(gray - blur3(gray, 2))
    contrast_white = white & (highpass > 0.05)
    height = gray.shape[0]
    top = contrast_white[: int(height * 0.65), :]
    return [
        float(contrast_white.mean()),
        float(top.mean()),
        float(contrast_white.sum() / max(1, white.sum())),
        float(white.mean()),
    ]


def extract_features(path):
    rgb = load_image(Path(path))
    gray, c1, c2 = gray_chroma(rgb)
    tiles, patch = select_tiles(gray)
    tile_scores = np.asarray([[t[0], t[3], t[4], t[5]] for t in tiles], dtype=np.float64)

    features = []
    features.extend(fft_features((gray, c1, c2), tiles, patch))
    features.extend(global_texture_features(gray, c1, c2))
    features.extend(stripe_features(gray, c1, c2))
    features.extend(border_features(gray))
    features.extend(cursorish_features(gray, rgb))
    features.extend(tile_scores.mean(axis=0))
    features.extend(tile_scores.max(axis=0))
    features.extend([float(len(tiles)), float(patch), float(gray.shape[0]), float(gray.shape[1])])
    return np.asarray(features, dtype=np.float64)


def train_logistic(z, y, l2=10.0, steps=1300, lr=0.08):
    weights = np.zeros(z.shape[1], dtype=np.float64)
    bias = 0.0
    n = float(len(y))

    for step in range(steps):
        pred = sigmoid(z @ weights + bias)
        eta = lr / (1.0 + step / 550.0)
        weights -= eta * (((z.T @ (pred - y)) / n) + (l2 * weights / n))
        bias -= eta * float((pred - y).mean())

    return weights, float(bias)
