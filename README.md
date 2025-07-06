# Custom Torch LSTM Model for Inferring Protein Diffusive Coefficients and States

## Overview

### Task
Given noisy protein coordinates in 2-dimensions, return the diffusion coefficient, anomalous exponent, and protein state. This can change at arbitrary points throughout the time series.

### Description
This project is for the Anomalous Diffusion Challenge 2024. It is a custom 3-stacked bidirectional LSTM model (with skip connections and dropout) developed by myself to infer diffusion coefficients, anomalous exponents, and states for noisy protein trajectories (fractional Brownian motion) undergoing an arbitrary number of changepoints. This is common when dealing with single molecule tracks from super-resolution fluorescence microscopy and informs us of changes in behaviour (e.g., confinement, clustering, directed motion, etc.). Features were selected from an extensive literature review and forward feature selection. Model finished in the Top 5.

### Variables of Interest
This project is particularly interested in determining:
- $K$ (diffusion coefficient)
- $\alpha$ (anomalous exponent)
- Protein state:
  - 0 - trapped
  - 1 - confined
  - 2 - freely diffusing
  - 3 - directed motion

$K$ and $\alpha$ are related through the mean square displacement (MSD, 2-D) given by:

$$ MSD = 4Kt^\alpha $$

### References
For more information on the challenge please refer to:

```python
G. Muñoz-Gil, H. Bachimanchi ...  C. Manzo
In-principle accepted at Nature Communications (Registered Report Phase 1)
arXiv:2311.18100
https://doi.org/10.48550/arXiv.2311.18100
```

## Project Structure

Feel free to download the code and run the models as notebooks.  
Training time varies depending on dataset size and GPU.  
For this work, we simulate ~5 million tracks, taking about **10 hours** on a single **NVIDIA RTX A5000**, with ~44 iterations/sec using a batch size of 32. Convergence typically occurs after ~20 epochs.

---

### `/simulations`

Generates simulated protein tracks using [AnDi](https://github.com/AnDiChallenge/andi_datasets):
- Uses multiprocessing for faster simulation.
- Tracks are saved individually to avoid memory issues.
- Files are then concatenated using `concat.py`.
- For faster training, tracks are converted into pickled dataset classes (`pickle_data.py`).
- Includes a commented `TimeSeriesDataset` class for on-the-fly training from individual files if data does not fit into memory (6x slower).
- You may provide your own dataset and skip this step.
- Alternatively, use provided model weights to skip training.

---

### `/main`

All training and inference scripts:
- Includes notebooks for:
  - `train_alpha`
  - `train_k`
  - `train_state`
  - `inference`
- Written as notebooks to ease experimentation and understanding.
- `inference.ipynb` includes changepoint detection using [ruptures](https://github.com/deepcharles/ruptures) with a penalty (number of changepoints is unknown).
- A notebook for hyperparameter tuning (learning rate, L2 lambda, batch size, dropout, model layers) using [optuna](https://github.com/optuna/optuna) is also included.
  - Currently optimises learning rate and L2 lambda.
  - Easily extendable to other parameters.

---

### `/main/models`

Models used for training:
- Handles both regression (K, alpha) and classification (state).
- Uses the same base model 3 times with different output layers.
  - Total trainable parameters: ~512k × 3 ≈ **1.5M**
- Optionally, a combined model returning `[K, alpha, state]` can be used.
  - Requires weighted loss and careful tuning.
- Multiple architectures (e.g., LSTM+CNN, Transformer+Attention) were tested.
- Final model was selected based on performance.
- Layer stacking continued until no further improvement (guided by [optuna](https://github.com/optuna/optuna)).
- Model input: (x, y) coordinates per timestep.
- Model output: timeseries of same length for the variable.
- If only interested in one variable (e.g., $K$), train only the relevant model.
- Notebooks are self-contained for modular use, at the cost of some code duplication.

---

### `/main/utils`

Utility functions used during training and inference:
- Feature extraction
- Post-processing
- Dataset class
- Plotting tools

`utils/plotting.py`:
- Generates multiple `.svg` figures for high-quality downstream editing (e.g., in Inkscape).

---

### `/plots`

- Destination folder for saving and editing generated figures.

---

### `/misc`

- Miscellaneous exploratory scripts and code.

---

> **Note:** Some changes will be made to the code for tidying and clarity.


## Requirements 
```
pandas
pyarrow
numpy
scikit-learn
andi-datasets
IPython
pylops (optional)
ruptures
scipy
torch
tensorboard
torchinfo
tqdm
matplotlib
torchmetrics
optuna (optional)
```

