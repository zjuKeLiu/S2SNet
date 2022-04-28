# Visualization Part
This folder is the explainable part of paper "S2SNet: A Pretrained Neural Network for Superconductivity Discovery", which mainly consists of following codes:

### 1. spGrad_output.ipynb
  Code for computing the gradients for moleculars. 
    
  Input: S2SNet model, moleculars' atoms and coordinates, lattice data and labels.  
  Output: gradients for moleculars' atoms and lattices.

### 2. Visualization.ipynb
  Visualize element importance, granidents' distributions and model's attention for some moleculars according to the results from spGrad_output.ipynb.  
    
  Input: The index of molecular want to be visualized.  
  Output: gradient heatmap of the molecular.