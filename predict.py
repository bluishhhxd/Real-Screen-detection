import json
import sys
from pathlib import Path

import numpy as np

from screen_features import extract_features, sigmoid


def load_model(model_path=None):
    if model_path is None:
        model_path = Path(__file__).with_name("screen_model.json")
    else:
        model_path = Path(model_path)

    with model_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_proba(image_path, model_path=None):
    model = load_model(model_path)
    features = extract_features(image_path)

    expected = int(model["feature_count"])
    if len(features) != expected:
        raise RuntimeError(f"Feature mismatch: got {len(features)}, expected {expected}")

    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    selected = np.asarray(model["selected"], dtype=int)
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])

    z = (features - mean) / scale
    return float(sigmoid(z[selected] @ weights + bias))


def main(argv):
    if len(argv) not in (2, 3):
        print("Usage: python predict.py image.jpg [screen_model.json]", file=sys.stderr)
        return 2

    image_path = argv[1]
    model_path = argv[2] if len(argv) == 3 else None
    probability = predict_proba(image_path, model_path)
    print(f"{probability:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
