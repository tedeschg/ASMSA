# ASMSA: Analisys and Sampling of Molecular dynamic Simulation with Adversarial autoencoder

ASMSA is an Adversarial Autoencoder designed to analyze molecular dynamics (MD) simulations by learning a compact, low-dimensional representation that captures the essential physical and chemical properties of the system.
In this latent space, spatial proximity corresponds to similarity in molecular behavior or characteristics, allowing for meaningful semantic analysis of complex simulations.

Moreover, the learned latent vectors provide an optimal set of collective variables (CVs) that can be directly employed to enhance sampling methods such as metadynamics.

The complete workflow is organized into a series of Jupyter notebooks, guiding the user through:

* Preprocessing and managing custom datasets,

* Training and fine-tuning model parameters,

* Extracting interpretable and task-specific embeddings.

This modular structure allows users to customize the pipeline according to their data and research goals.

## Badges
TODO

## Getting started

Full support for distributed hyperparameter tuning is available at CERIT-SC Jupyterhub:

1. Go to https://hub.cloud.e-infra.cz/ and log in with your [Metacentrum](http://metacentrum.cz) account
1. Either click on **Start my server** for the default, or type a specific name and click **Add New Server**
1. Fill in the submit form:
    - Select an image: **Custom**
    - Custom image name: **cerit.io/ljocha/asmsa:2025-1**
    - Select persistent home type: **New** at the first time, **Existing** is prefered afterwards
    - Select persistent home (when *Existing* in the previous choice): pick you prefered one, or stick with the only one offered
    - I want MetaCentrum home: **no**
    - Would you like to connect project directory: **no**
    - Select number of CPUs: **2** (it's enough for the notebook, parallel workers are not counted here)
    - Memory: **8 GB** (appears to be enough for this usecase)
    - GPU: **None** (it turns out that our models are too small to leverage GPU accelleration)
1. Click on **Start**
1. Depending on the container image cache status Jupyterlab starts in few seconds (the image was cached) or several minutes (it must be downloaded)
1. click on **asmsa.ipynb** in the left panel and follow instructions in the notebook

 
## Visuals
TODO (screenshots, ...)

## Installation
TODO

## Usage
### 1. prepare.ipynb
### 2. tune.ipynb
### 3. train.ipynb
### 4. plumed.ipynb
### 5. md.ipynb

## Support
TODO

## License
TODO

## Project status
In development... No main usable version released yet

