# Custom Torch LSTM Model for Inferring Protein Diffusive Coefficients and States

## Overview 
Task: Given noisy protein coordinates in 2-dimensions, return the diffusion coefficient, anomolous exponent, and protein state. This can change at arbitrary points throughout the time series.

This project is for the Anomalous Diffusion Challenge 2024. It is a custom 3-stacked bidirectional LSTM model (with skip connections and dropout) developed by myself to infer diffusion coefficients, anomalous exponents, and states for noisy protein trajectories (fractional Brownian motion) undergoing an arbitrary number of changepoints. This is common when dealing with single molecule tracks from super-resolution fluorescence microscopy and informs us of changes in behaviour (e.g., confinement, clustering, directed motion, etc.). Features were selected from an extensive literature review and forward feature selection. Model finished in the top 5.

This project is particulary interested in determining $K$ (diffusion coefficient), $\alpha$ (anomalous exponent), and the protein state (0 - trapped, 1 - confined, 2 - freely diffusing, 3 - directed motion). $K$ and $\alpha$ are related through the mean square displacement (MSD, 2-D) given by $MSD = 4Kt^\alpha$

For more information on the challenge please refer to:

``` python
G. Muñoz-Gil, H. Bachimanchi ...  C. Manzo
In-principle accepted at Nature Communications (Registered Report Phase 1)
arXiv:2311.18100
https://doi.org/10.48550/arXiv.2311.18100
```
or [Anomalous Diffusion GitHub page](https://github.com/AnDiChallenge/andi_datasets)


## Usage

Feel free to download the code and run the models as notebooks.
Training time varies depending on dataset size and GPU (for this work we simulate ~5 million tracks taking about a ~10hrs on a single NVIDIA RTX A5000, ~44 it/s with batch size 32 and convergence after ~20 epochs).

**/simulations** - Generates simulated protein tracks using [AnDi](https://github.com/AnDiChallenge/andi_datasets) (uses multiprocessing to do this faster). The tracks are saved individually to avoid memory issues and it breaking. They are then concatenated (concat.py). For faster training, the dataset class for the protein tracks are pickled (pickle_data.py). There is another TimeSeriesDataset implementation commented in case you want to train from individual files but this is slower (roughly 6x). You can also provide your own dataset and ignore this. Alternatively, use provided model weights if you want to skip training. <br>

**/main** - All the training and inference file (train_alpha, train_k, train_state, inference). These are as notebooks to make it easy to follow and since I also did a lot of experimenation. The inference.ipynb includes changepoint detection using [ruptures](https://github.com/deepcharles/ruptures) with a penalty since the number of changepoints is unknown. There is also a notebook for hyperparameter tuning for learning rate, L2 lambda, batch size, dropout, model layers, etc. using [optuna](https://github.com/optuna/optuna).  Right now this shows the learning rate and L2 lambda optimisation but can easily be adapted to the others using the docs. <br>

**/main/models** - Models used. Since we are performing both regression (2 variables - K and alpha) and classification (1 variable - state), I have made use of the same model 3 times with a different output layer for state (total trainable parameters ≈ 512k * 3 ≈ 1.5M). The model can potentially be combined into a single model returning a vector with [K, alpha, state]. This was experimented with but requires careful tuning of a weighted loss function and is more difficult. Various architectures (e.g. LSTM+CNN, Transformer+Attention, etc.) were explored and custom loss functions but the model in this folder worked best. Layers were added to the model until no improvement was seen (using [optuna](https://github.com/optuna/optuna)). The model only needs (x,y) coordinates for every timestep as input and extracts the other features. The output is a timeseries of the same length for the variable in question. If you do not need some output labels, do not use/train those models (e.g. only care about $K$). Every training notebook is stand-alone at the expense of repeated code but it makes it easier to follow and in case some are only interested in a subset of variables.<br>

**/main/utils** - Various functions used throughout training and inference e.g., feature extraction, post-processing, time series dataset class, and plotting. It needs to be tidied up a bit. utils/plotting.py creates a lot of svg files for downstream editing in Inkscape for high quality figures <br>

**/plots** - location to save figures for visualising and editing

**/misc** - miscellaneous exploratory code

**Note:** Some changes will be made to code for tidying/clarity

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

