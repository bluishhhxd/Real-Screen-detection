# Spot the Fake Photo

Detect whether a single image is a real photo or a photo of a screen.

`predict.py` prints a number from `0` to `1`:

- `0` means likely real photo
- `1` means likely photo-of-a-screen
- `>= 0.5` is classified as screen

## Setup

Create/activate a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

This project only needs:

- NumPy
- Pillow

## Predict One Image

Put your image in the project folder, then run:

```powershell
python predict.py test.jpg
```

Example output:

```text
0.873421
```

That means the model thinks the image is likely a photo of a screen.

If the filename has spaces:

```powershell
python predict.py "my test photo.jpg"
```

## Train Again

The dataset should have this structure:

```text
dataset/
  real/
    1.jpg
    ...
  screen/
    1.jpg
    ...
```

Run:

```powershell
python train.py --dataset dataset --output screen_model.json
```

This extracts features, runs pair-aware validation, and writes the model used by `predict.py`.

## Approach

I used a small classical computer-vision pipeline instead of a neural network. Photos of screens often contain artifacts that real photos do not: display pixel-grid texture, moire patterns, row/column stripe energy, RGB subpixel color aliasing, and sometimes dark screen borders or cursor-like details.

The model extracts those handcrafted features with NumPy/Pillow and feeds them into a tiny logistic regression model saved in `screen_model.json`.

## Results

On the local 100-image dataset:

- Pair-aware cross-validation accuracy: `95/100 = 95.0%`
- Final training accuracy: `99/100 = 99.0%`
- Warm-process latency: about `838 ms/image`
- Fresh CLI latency: about `945 ms/image`
- Cost per image: effectively free on-device

More detail is in `NOTES.md`.
