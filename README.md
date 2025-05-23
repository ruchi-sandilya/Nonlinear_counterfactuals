# Nonlinear Counterfactual Generation

This repository contains Jupyter notebooks that demonstrate three nonlinear counterfactual generation approaches: **FLEXD**, **SPLINED**, and **DICED**. Each method enables controllable trajectory generation in latent space for counterfactual analysis.

### Notebooks Included

1. **FLEXD_vs_Baselines**  
   Compares FLEXD with the baseline methods Direct, Lerp, Slerp, and LSTM using DDIM latent representations from fluid dynamics simulations.

2. **SPLINED_vs_Baselines**  
   Compares SPLINED with Lerp, Slerp, and LSTM using CEBRA embeddings of calcium imaging data.

3. **Counterfactual_Explainability_DICED**  
   Demonstrates how to identify counterfactual directions between responder and non-responder groups in TMS-induced E-field embeddings using KDE.

---

## Data Representations by Approach

- **FLEXD** uses DDIM latent vectors from fluid datasets.
- **SPLINED** uses CEBRA-embedded DDIM latents from calcium imaging data.
- **DICED** uses CEBRA-embedded E-field DDIM latents for KDE-based counterfactual explanation.

---

## Dataset Access

The fluid dynamics dataset used in these analyses can be downloaded from the following link:  
[https://wcm.box.com/s/tlul4yndcbluohc9gcmilk6pn6kcoldw](https://wcm.box.com/s/tlul4yndcbluohc9gcmilk6pn6kcoldw)

---

## Notebook Descriptions

### 1. `FLEXD_vs_Baselines.ipynb`

**Purpose**:  
Learn local derivatives in DDIM latent space to predict future latents using FLEXD. This notebook also compares FLEXD to Direct, Lerp, Slerp, and LSTM baselines.

**Key Features**:
- Predictive counterfactual generation using FLEXD
- Counterfactual synthesis using a pretrained conditional Latent Diffusion Model (cLDM)
- Visual and quantitative comparisons with baseline methods

---

### 2. `SPLINED_vs_Baselines.ipynb`

**Purpose**:  
Embed high-dimensional DDIM latents (with shape `4 × 32 × 32`) into a 3D space using CEBRA. Fit a smooth cubic B-spline to the embedded trajectory and project it back to the DDIM space using KNN interpolation.

**Key Features**:
- CEBRA-based embedding that preserves local distances
- Spline-based interpolation for smooth trajectory generation
- Reconstruction of modified latents using KNN-based projection and pretrained cLDM

---

### 3. `Counterfactual_Explainability_DICED.ipynb`

**Purpose**:  
Embed DDIM latents of TMS-induced E-field data for responders and non-responders. Use a nonlinear RBF-SVM for classification in the embedded space. Apply KDE to each class to identify high-density peaks. Generate counterfactuals by interpolating between these peaks and projecting the result back to the DDIM space.

**Key Features**:
- Nonlinear classification in embedded space
- KDE-based analysis for identifying regions of interest
- Counterfactual generation through density-informed interpolation

---

## Dependencies

The following Python libraries are required to run the notebooks:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `jupyter`
- `cebra`

### Installation

Install all dependencies with:

```bash
pip install numpy scipy matplotlib scikit-learn jupyter cebra
