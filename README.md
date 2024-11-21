# Nonlinear Counterfactual Generation

This repository contains Jupyter notebooks focused on nonlinear counterfactual generation using two approaches:  
1. **Embedding and moving along a principal curve**  
2. **Embedding and kernel density estimation (KDE)**  

We demonstrate these methods using:  
- **DDIM latent vectors** of fluid datasets for the principal curve approach.  
- **E-Field datasets** of responders and non-responders for the KDE approach.

### Dataset
The datasets used in these analyses can be downloaded from the following link:  
[https://wcm.box.com/s/tlul4yndcbluohc9gcmilk6pn6kcoldw](https://wcm.box.com/s/tlul4yndcbluohc9gcmilk6pn6kcoldw)


## Notebooks

### 1. **Embedding-and-moving-along-principal-curve.ipynb**
   - **Purpose**: 
     - Embeds DDIM latent data of dimension ($4 \times 32 \times 32$) into 2D space using CEBRA, which approximately preserves neighbor distances.
     - Fits an elastic principal curve to the embedded DDIM latent vectors.
     - Generates counterfactuals by traversing along the estimated principal curve and modifying latent vectors.
   - **Key Features**:
     - Embedding DDIM latent space in a way that locally preserves distances.
     - Fitting a principal curve to the embedded DDIM latent space.
     - Estimating the principal curve using pairwise distances between consecutive nodes of the curve.
     - Modifying embedded latents by moving along these curves.
     - Projecting modified embedded latents back to DDIM space using KNN interpolation.
     - Generating counterfactuals using modified DDIM latents and a pretrained cLDM.

### 2. **Embedding_and_KDE.ipynb**
   - **Purpose**: 
     - Embeds E-Field DDIM latents of responder and non-responder patients into a reduced-dimensional space.
     - Applies Kernel Density Estimation (KDE) to visualize and analyze the probability density of data in the embedded space.
     - Identifies peak points for responders and non-responders.
     - Transforms a non-responder into a responder by leveraging the peak differences.
   - **Key Features**:
     - Dimensionality reduction techniques for embedding.
     - Visualization and density analysis using KDE.
     - Identification of responder and non-responder peaks.
     - Transformation of non-responder profiles to responder profiles using KDE peak differences.

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
