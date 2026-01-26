# SCANNER

Implementation of the paper "SCANNER: A Spatio-temporal Correlation and Neighborhood-based Feature Enrichment for Traffic Prediction", accepted as a poster at ACM SIGSPATIAL 2023.

## Summary
In this repository, you can find the code to train and evalute SCANNER, a novel approach analyzing the correlation between road segments in different spatio-temporal contexts to capture patterns such as the speed change propagation across road segments. 

## Code Repository

The official GitHub repository for this project is:

* **[https://github.com/D-Stiv/SCANNER](https://github.com/D-Stiv/SCANNER)**


## Tool Version

This repository corresponds to:

* **SCANNER v1.0.0** (initial research release accompanying the SIGSPATIAL publication)

Future updates or experimental extensions should be versioned separately.


* **Operating System**: Any 64-bit OS supporting Python and PyTorch
  (e.g., Linux, macOS, Windows)
* **Python**: 3.8.30
* **PyTorch**: 1.9.0+cu111
* **NumPy**: 1.23.5
* **Architecture**: 64-bit machine
* **Containerization**:
  No container (e.g., Docker) is required. The code runs in a standard Python virtual environment.

GPU support is optional but recommended for faster training.

## Setup
To create the correlation matrix
```bash
python3 create_correlation_natrix.py
```

Create and activate a virtual environment:

```bash
virtualenv venv
source venv/bin/activate
```

Install PyTorch, PyTorch Geometric, and remaining dependencies:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install torch-geometric
pip install -r requirements.txt
```

## Prerequisites

1. Compute the temporal correlation matrices **B** and save them in `.pkl` format.

   * Shape: `L × N × N`
   * `L`: number of temporal lags
   * `N`: number of nodes

2. Save the spatial distance matrix **A** in `.pkl` format.

   * Shape: `N × N`

3. Save the dataset **W** in `.h5` format.

   * Shape: `T_max × N`
   * The index of **W** must be castable to `datetime`.

4. Update file paths in the source code

## Run the Code

Configuration parameters are defined in `run.py`.

Example command using all default parameters
```bash
python3 run.py
```

## Citation
If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@InProceedings{gounoueGIS23,
  author    = {Steve Gounoue and Ran Yu and Elena Demidova},
  booktitle = {The 31st ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '23), November 13--16, 2023,Hamburg, Germany},
  title     = {SCANNER: A Spatio-temporal Correlation and Neighborhood-based Feature Enrichment for Traffic Prediction},
  year      = {2023},
  doi       = {10.1145/3589132.3625653},
  isbn      = {979-8-4007-0168-9/23/11},
}
```


## License
This repository contains code from other sources under ``MIT License``

We re-implement the spatio-temporal normalization from the paper
"ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting"
(https://doi.org/10.1145/3447548.3467330)

We extend Graph-Wavenet (Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019), inserting the spatio-temporal normalization (``gwnet`` in ``run.py``).

Modifications are indicated in individual file header.