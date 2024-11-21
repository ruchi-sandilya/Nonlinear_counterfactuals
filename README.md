# Nonlinear counterfactual generation
This repository contains Jupyter notebooks focused on nonlinear counterfactual generation using two approaches: 1. Embedding and moving along a principal curve and 2. Embedding and kernel density estimation (KDE). We demonstrate the method using fluid datasets (for principal curve approach) and E-Field datasets of responders and non-responders patient for KDE appoach.

## Notebooks
### 1. **Embedding-and-moving-along-principal-curve.ipynb**
   - **Purpose**: 
     - Explores data embedding and analyzes the trajectory of data along principal curves.
     - Provides a method to uncover the underlying structure in high-dimensional data.
   - **Key Features**:
     - Principal curve fitting and visualization.
     - Insights into data patterns by moving along these curves.
     - Useful for exploring trends and clusters in complex datasets.
### 2. **Embedding_and_KDE.ipynb**
   - **Purpose**: 
     - Demonstrates embedding data into a lower-dimensional space.
     - Applies Kernel Density Estimation (KDE) to visualize and analyze the probability density of data in the embedded space.
   - **Key Features**:
     - Implementation of dimensionality reduction techniques.
     - Visualization of data density using KDE.
     - Practical examples of embedding workflows for structured data.



## Dependencies
The following Python libraries are required to run these notebooks:
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `sklearn` (scikit-learn)
- `pandas`
- `jupyter`

You can install these dependencies using:
```bash
pip install numpy scipy matplotlib seaborn scikit-learn pandas jupyter
