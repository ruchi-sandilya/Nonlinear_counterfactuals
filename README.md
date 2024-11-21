# Nonlinear counterfactual generation
This repository contains Jupyter notebooks focused on nonlinear counterfactual generation using two approaches: 1. Embedding and moving along a principal curve and 2. Embedding and kernel density estimation (KDE). We demonstrate the method using DDIM latent vectors of fluid datasets (for principal curve approach) and E-Field datasets of responders and non-responders patient for KDE appoach.

## Notebooks
### 1. **Embedding-and-moving-along-principal-curve.ipynb**
   - **Purpose**: 
     - Embeds DDIM latent data of dimension ($4\times 32 \times 32$) into 2D space using CEBRA that approximately preserves neighbor distances.
     - Fits elastic principal curve to the embedded DDIM latent vectors.
     - Generates counterfactuals by modifying latent by traversing along the estimated principal curve
   - **Key Features**:
     - Embedding DDIM latent space in local distance preserving way.
     - Principal curve fitting and of embedded DDIM latent space.
     - Estimating principal curve using pairwise distances between consecutive nodes of the curve.
     - Modifying embedded latents by moving along these curves.
     - Projecting modified embedded latents back to DDIM space using KNN interpolation.
     - Generating counterfactuals using modified DDIM latents and pretrained cLDM.
### 2. **Embedding_and_KDE.ipynb**
   - **Purpose**: 
     - Demonstrates the embedding of E-Field DDIM latents of responder and non-resonder patients.
     - Applies Kernel Density Estimation (KDE) to visualize and analyze the probability density of data in the embedded space.
     - Identifies responders and non-responder peak points
     - Transforms a non-responder to responder using peak difference
   - **Key Features**:
     - Implementation of dimensionality reduction techniques.
     - Visualization of data density using KDE.
     - Practical examples of embedding workflows for structured data.

## Dependencies
The following Python libraries are required to run these notebooks:
- `numpy`
- `scipy`
- `matplotlib`
- `sklearn` (scikit-learn)
- `jupyter`
- `cebra`

You can install these dependencies using:
```bash
pip install numpy scipy matplotlib scikit-learn jupyter cebra
