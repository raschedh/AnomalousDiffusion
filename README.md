# Custom Torch LSTM Model for Inferring Protein Diffusive Coefficients and States

## Overview 

Author: Rasched Haidari

This project is for the Anomalous Diffusion Challenge 2024. It is a custom 3-stacked LSTM model with skip connections (PyTorch) developed by myself to infer diffusion coefficients, anomalous exponents and states for noisy protein trajectories undergoing an arbitrary number of changepoints. This is common when dealing with single molecule tracks from super-resolution fluorescence microscopy and informs us of changes in behaviour (e.g., confinement, clustering, directed motion, etc.). Features were selected from an extensive literature review and feature selection (forward selection).

This project is particulary interested in determining $K$ (diffusion coefficient), $\alpha$ (anomalous exponent), and the protein state (0 - trapped, 1 - confined, 2 - freely diffusing, 3 - directed motion). $K$ and $\alpha$ are related through the mean square displacement (MSD, 2-D) given by $MSD = 4Kt^\alpha$

For more information on the challenge please refer to:

``` python
G. Muñoz-Gil, H. Bachimanchi ...  C. Manzo
In-principle accepted at Nature Communications (Registered Report Phase 1)
arXiv:2311.18100
https://doi.org/10.48550/arXiv.2311.18100
```
or [Anomalous Diffusion GitHub page](https://github.com/AnDiChallenge/andi_datasets)

Model architecture was optimised using Optuna and data was simulated from the AnDi Python package.

## Usage

**/data** - Generates dataset used for training, validation and testing<br>

**/models** - Models used. Since we are performing both regression (2 variables - K and alpha) and classificaiton (1 variable - state), I have duplicated the model 3 times with a different output layer for state. The model can potentially be combined into a single model returnining a vector with (K, alpha, state). This was experimented with but requires careful tuning of a weighted loss function and is more difficult. For the sake of simplicity and due to time-constraints, this was not implemented.<br>

**/src** - All the training and inference files. These are as notebooks to make it easy to follow. The inference files includes changepoint detection using ruptures with a penalty since the number of changepoints is unknown. <br>

**/utils** - Various functions used throughout training and inference.<br>

**/optuna** - Hyperparameter tuning<br>


