# Code Documentation

This repository provides the important code files as part of the Master's Thesis "Physics-driven data generation for strain-based damage detection on the example of an aerospace sandwich structure", written at the *Institute of Structural Lightweight Design* at the JKU Linz.
Its sole purpose is to serve as a documentation repository, it is not meant to provide a fully reproducible version of the whole project.
Therefore, this repository is archived, making it read-only. The code can still be downloaded.

## Content structure
### models/FEM_model
This directory holds all relevant code files regarding the creation and simulation of submodels and global loadcases.
It contains *module* files, as well as Abaqus *scipts*.
In general, the specific Abaqus project of the submodel is necesseray to fully execute these files.

### models/framework
In this directory all code of the framework implementation is stored.
The main file is `framework_main.py`. The remaining files provide support functions.

### notebooks
The data analysis and development of SHM-methods is mainly done in Jupyter Notebooks.
These heavily rely on support files, which store the detailed implementations of classifier-classes, plotting utilities, etc.

### data/raw
The data used for training and testing and which is also visualized is provided in the form of `.csv` files.
