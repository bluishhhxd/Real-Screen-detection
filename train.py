import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from screen_features import (
    MAX_SIDE,
    MAX_TILES,
    PATCH_SIZE,
    extract_features,
    sigmoid,
    train_logistic,
)


DEFAULT_K = 170
DEFAULT_L2 = 10.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def numeric_id(path):
    match = re.search(r"\d+", path.stem)
    return int(match.group(0)) if match else path.stem


def collect_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    samples = []
    for class_name, label in (("real", 0), ("screen", 1)):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing folder: {class_dir}")

        paths = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        paths.sort(key=lambda path: (numeric_id(path), path.name))
        for path in paths:
            samples.append((path, label, numeric_id(path)))

    if not samples:
        raise RuntimeError(f"No images found under {dataset_dir}")
    return samples


def feature_matrix(samples):
    rows = []
    started = time.perf_counter()
    for index, (path, _, _) in enumerate(samples, 1):
        print(f"[{index:03d}/{len(samples):03d}] extracting {path}")
        rows.append(extract_features(path))
    elapsed = time.perf_counter() - started
    return np.vstack(rows), elapsed


def standardize_train_apply(x_train, x_apply):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) + 1e-6
    return (x_train - mean) / scale, (x_apply - mean) / scale, mean, scale


def select_features(z_train, y_train, k):
    pos = z_train[y_train == 1]
    neg = z_train[y_train == 0]
    scores = np.abs(pos.mean(axis=0) - neg.mean(axis=0)) / (z_train.std(axis=0) + 1e-6)
    return np.argsort(scores)[::-1][:k]


def fit_model(x, y, k=DEFAULT_K, l2=DEFAULT_L2):
    z, _, mean, scale = standardize_train_apply(x, x)
    selected = select_features(z, y, k)
    weights, bias = train_logistic(z[:, selected], y, l2=l2)
    return {
        "version": "screen-grid-logreg-v1",
        "max_side": MAX_SIDE,
        "patch_size": PATCH_SIZE,
        "max_tiles": MAX_TILES,
        "feature_count": int(x.shape[1]),
        "k": int(k),
        "l2": float(l2),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "selected": selected.astype(int).tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
    }


def predict_with_fold(x_train, y_train, x_test, k, l2):
    z_train, z_test, _, _ = standardize_train_apply(x_train, x_test)
    selected = select_features(z_train, y_train, k)
    weights, bias = train_logistic(z_train[:, selected], y_train, l2=l2)
    return sigmoid(z_test[:, selected] @ weights + bias)


def cross_validate(x, y, pair_ids, paths, k=DEFAULT_K, l2=DEFAULT_L2):
    probs = np.zeros(len(y), dtype=np.float64)
    unique_ids = sorted(set(pair_ids), key=str)

    for pair_id in unique_ids:
        test_mask = np.asarray([pid == pair_id for pid in pair_ids])
        train_mask = ~test_mask
        probs[test_mask] = predict_with_fold(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            k=k,
            l2=l2,
        )

    pred = (probs >= 0.5).astype(int)
    wrong = [
        {
            "path": str(paths[i]),
            "label": int(y[i]),
            "prob_screen": float(probs[i]),
        }
        for i in np.where(pred != y)[0]
    ]
    return {
        "accuracy": float((pred == y).mean()),
        "wrong_count": int(len(wrong)),
        "wrong": wrong,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="screen_model.json")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--l2", type=float, default=DEFAULT_L2)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    samples = collect_dataset(args.dataset)
    paths = [sample[0] for sample in samples]
    y = np.asarray([sample[1] for sample in samples], dtype=int)
    pair_ids = [sample[2] for sample in samples]

    x, feature_seconds = feature_matrix(samples)
    print(
        f"Extracted {x.shape[1]} features for {len(samples)} images "
        f"in {feature_seconds:.2f}s ({feature_seconds * 1000 / len(samples):.1f} ms/image)."
    )

    cv = None
    if not args.skip_cv:
        started = time.perf_counter()
        cv = cross_validate(x, y, pair_ids, paths, k=args.k, l2=args.l2)
        cv["seconds"] = float(time.perf_counter() - started)
        print(
            f"Pair-aware CV accuracy: {cv['accuracy']:.3f} "
            f"({len(samples) - cv['wrong_count']}/{len(samples)})"
        )
        if cv["wrong"]:
            print("Wrong CV predictions:")
            for item in cv["wrong"]:
                print(
                    f"  {item['path']} label={item['label']} "
                    f"prob_screen={item['prob_screen']:.3f}"
                )

    model = fit_model(x, y, k=args.k, l2=args.l2)
    train_probs = predict_model_dict(model, x)
    model["training_accuracy"] = float(((train_probs >= 0.5).astype(int) == y).mean())
    model["training_image_count"] = int(len(samples))
    model["training_feature_ms_per_image"] = float(feature_seconds * 1000 / len(samples))
    if cv is not None:
        model["pair_cv"] = cv

    output = Path(args.output)
    output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved {output}")


def predict_model_dict(model, x):
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    selected = np.asarray(model["selected"], dtype=int)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    z = (x - mean) / scale
    return sigmoid(z[:, selected] @ weights + bias)


if __name__ == "__main__":
    main()
