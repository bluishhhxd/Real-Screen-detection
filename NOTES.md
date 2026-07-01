# Spot the Fake Photo - Approach Note

## Approach

I used a small classical CV pipeline instead of a neural net. A photo of a laptop screen usually carries artifacts that a direct real-world photo does not: pixel-grid/moire texture, periodic row/column energy, color-channel aliasing from RGB subpixels, and sometimes display context such as dark bezels or a mouse cursor. `screen_features.py` extracts those signals with Pillow and NumPy only:

- FFT peak/axis energy from low-edge image tiles.
- Global high-frequency luma/chroma texture.
- 1D row/column stripe spectra.
- Border darkness and simple cursor-like bright-contrast cues.

`train.py` standardizes the features, selects the strongest 170 features on the training split, then fits a tiny L2-regularized logistic regression implemented directly in NumPy. `predict.py` loads `screen_model.json`, extracts features for one image, and prints a probability from 0 to 1 where 1 means photo-of-screen.

## Validation

I validated with leave-one-pair-out cross-validation: for image id `N`, both `real/N` and `screen/N` are held out together, so the model never trains on the same underlying scene it is testing.

- Dataset: 50 real, 50 screen.
- Pair-aware CV accuracy: 95/100 = 95.0%.
- Final model training accuracy on all 100 images: 99/100 = 99.0%.

The remaining CV mistakes were mostly very blurred screen photos where the screen grid is weak, plus a couple of real photos with dark borders/texture that look screen-like.

## Latency and Cost

Measured in the local Codex Windows workspace using Python 3.12.13, NumPy 2.3.5, Pillow 12.2.0:

- Warm Python process: mean 837.9 ms/image, median 886.2 ms/image over 8 images.
- Fresh CLI call (`python predict.py image.jpg`): mean 945.2 ms/image over 3 images.
- Cost per image on-device: effectively free.
- Rough cloud cost: on a $0.05/hour CPU VM at 0.84 s/image, about $0.012 per 1,000 images or about $12 per million images. A faster CPU or batching multiple images per process would reduce this.

## Improvements

The biggest improvement would be collecting a more varied training set: different screens, phones, brightness levels, distances, and crops where the laptop bezel is not visible. I would also optimize the feature extractor by computing only selected features and using OpenCV/Numba for the small convolutions and FFT prep if dependency size were allowed.
