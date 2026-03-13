# LiSAD

Official implementation of the paper **Physical Information Consistency for Sensing Attack Detection in Lithium-Ion Battery Management Systems**.


## Overall Framework

![Overall framework](figures/overall_framework.pdf)

Full-resolution figure: [figures/overall_framework.pdf](figures/overall_framework.pdf)

The LiSAD pipeline contains three main stages:

- physics-informed pretraining on normal battery measurements,
- construction of physical evidence vectors from consistency losses,
- unsupervised sensing attack detection and interpretable attribution in evidence space.

## Overview

LiSAD is a physics-informed sensing attack detection framework for lithium-ion battery management systems. This repository contains the code used to:

- preprocess battery pack data from FOBSS and EBAI-style sources,
- train a physics-informed neural model (`LiBPINN`) with consistency losses,
- extract physics residual features for attack detection,
- evaluate detection quality with both classification and delay-aware metrics.

The core idea is to constrain latent battery states and outputs with physical-information consistency, including:

- electrical consistency,
- thermal consistency,
- state-of-charge consistency,
- structural smoothness and monotonicity priors.

## Repository Structure

```text
LiSAD/
|-- config.py                  # Centralized experiment configuration
|-- datasets/                  # Data loading, alignment, attack synthesis
|   |-- attack.py
|   |-- dataloader.py
|   |-- fobss.py
|   `-- utils.py
|-- models/                    # LiBPINN backbone, residual losses, detector wrapper
|   |-- battery.py
|   |-- detector.py
|   |-- losses.py
|   |-- modules.py
|   `-- utils.py
|-- scripts/                   # Training entrypoint and utilities
|   |-- loss_balancer.py
|   |-- pretrain.py
|   `-- utils.py
|-- eval/                      # Evaluation metrics
|   `-- metrics.py
|-- checkpoints/               # Saved checkpoints
|-- data/                      # Local datasets
|-- figures/                   # Paper figures or exported visualizations
`-- results/                   # Experiment outputs
```

## Environment

The codebase is implemented in Python and depends on the following main packages:

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `pydantic`
- `rich`
- `einops`
- `pyod`

A minimal environment can be created with:

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn pydantic rich einops pyod
```

## Datasets

This repository currently supports two data sources:

- `FOBSS`: a public modular battery monitoring dataset,
- `EBAI`: a self-built dataset used in this project.

### Dataset access

- `FOBSS`: available at https://www.doi.org/10.5445/IR/1000094469
- `EBAI`: the raw data are not directly distributed in this repository; please contact the authors to request access to the original self-built dataset

### FOBSS layout

`datasets.fobss.FOBSS` expects a directory layout like:

```text
<root_dir>/
`-- data/
    `-- <profile_name>/
        |-- Battery_Current.csv
        |-- Battery_Voltage.csv
        |-- Inverter_Current.csv
        |-- Inverter_Voltage.csv
        |-- Slave_0_Cell_Temperatures.csv
        |-- Slave_0_Cell_Voltages.csv
        `-- ...
```

### EBAI layout

`datasets.dataloader.preprocess_ebai_data` expects csv files containing columns similar to:

- `timestamp`
- `current_mA`
- `cell1_mV ... cell9_mV`
- `cell1_C ... cell9_C`
- `env_C`

Place the files under a local folder such as `data/EBAI/raw/`.

## Training

The main training entrypoint is:

```bash
python scripts/pretrain.py
```

### Pretrain on FOBSS

```bash
python scripts/pretrain.py \
  --dataset fobss \
  --root_dir ./data/FOBSS/ \
  --foldername "all" \
  --n_cell 11 \
  --save_dir ./checkpoints/fobss/ \
  --epochs 50
```

### Pretrain on EBAI

```bash
python scripts/pretrain.py \
  --dataset ebai \
  --root_dir ./data/EBAI/raw/ \
  --filenames "BMS_Data_2026-01-16T14-38-45-380Z.csv,BMS_Data_2026-01-22T11-16-48-168Z.csv" \
  --n_cell 9 \
  --save_dir ./checkpoints/ebai/ \
  --batch_size 256 \
  --epochs 500 \
  --use_balancer \
  --use_scheduler \
  --window_length 100 \
  --step 1
```

### Important arguments

- `--dataset`: choose `fobss` or `ebai`
- `--root_dir`: dataset root path
- `--foldername`: FOBSS profile names, comma-separated or `all`
- `--filenames`: EBAI csv filenames, comma-separated
- `--n_cell`: number of cells per pack
- `--interval`: resampling interval in seconds
- `--window_length`: sliding-window length
- `--step`: sliding-window stride
- `--use_balancer`: enable covariance-based loss balancing
- `--use_scheduler`: enable `OneCycleLR`

Default values are defined in [config.py](/d:/Home/CodeSpace/LiSAD/config.py).

## Method Components

### `LiBPINN`

Implemented in [models/battery.py](/d:/Home/CodeSpace/LiSAD/models/battery.py), the model combines:

- a spatiotemporal backbone (`NodeAttnLSTM`),
- Bayesian positive physical parameters for the equivalent-circuit model,
- estimators for polarization voltage, SOC, and open-circuit voltage,
- a modular physics-informed residual loss block.

### Physics-informed losses

Implemented in [models/losses.py](/d:/Home/CodeSpace/LiSAD/models/losses.py), the training objective includes:

- `KVL_loss`
- `dVdt_loss`
- `RC1_loss`
- `RC2_loss`
- `dSOCdt_loss`
- `dTdt_loss`
- `mono_loss`
- `smooth_loss`
- `KL_loss`

### Attack detector

[models/detector.py](/d:/Home/CodeSpace/LiSAD/models/detector.py) wraps the pretrained PINN and converts residual losses into tabular anomaly features for a downstream `pyod` detector.

### Evaluation metrics

[eval/metrics.py](/d:/Home/CodeSpace/LiSAD/eval/metrics.py) provides:

- `AUC-ROC`
- `AUC-PR`
- `F1`
- `FPR@95TPR`
- `TTFD`
- `NWDD`
- `TAS`

## Outputs

Training produces:

- model checkpoints in `checkpoints/`,
- training histories in `logs/`,
- processed datasets in `data/`,
- evaluation tables in `results/`.

## Acknowledgements

This implementation builds upon ideas and software from the following open-source repositories:

- `pyod`: https://github.com/yzhao062/pyod
- `TSB-AD`: https://github.com/thedatumorg/TSB-AD

We thank the authors and contributors of these projects for making their code publicly available.

## Ongoing Updates

More experiment scripts, benchmark pipelines, and visualization utilities will be organized and uploaded in follow-up updates to this repository.


## License

This project is released under the license provided in [LICENSE](/d:/Home/CodeSpace/LiSAD/LICENSE).
