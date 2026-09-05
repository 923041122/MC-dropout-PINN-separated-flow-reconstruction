# MC-Dropout Physics-Constrained Reconstruction for Separated Flows

This repository contains the code and supporting data used for the manuscript:

**"Uncertainty Quantification for High-Reynolds-Number Cylinder-Wake Flow Using an MC-Dropout Physics-Constrained Reconstruction Model"**

Authors: **Linlin Zhu and Xiaobing Zhang**

Affiliations:

1. School of Energy and Power Engineering, Nanjing University of Science and Technology, Nanjing 210094, China
2. Chongqing Hongyu Precision Industry Group Co., Ltd., Chongqing 402760, China

Corresponding author: **Xiaobing Zhang**  
Email: **zhangxb680504@163.com**

## Scope

The repository implements the MC-dropout physics-constrained reconstruction framework described in the manuscript.

The method is intended for physics-constrained reconstruction and conditional uncertainty/reliability assessment of separated-flow reference fields.

It should **not** be interpreted as:

- a complete SST-RANS solver,
- a turbulence-resolving simulation method,
- or a full Bayesian turbulence model.

For the cylinder-wake case, the reference velocity and pressure fields originate from a k-omega SST CFD dataset. The physics loss uses simplified two-dimensional incompressible momentum residuals as soft physical regularization and does not contain the complete k-omega SST transport equations, eddy-viscosity closure, or modeled Reynolds-stress terms.

For the NASA wall-mounted hump case, LES mean-field and experimental data are used as reference information. The simplified momentum residual is likewise treated as soft physical regularization rather than as a closed LES mean-flow governing system.

## Repository structure

```text
.
├── cylinder_wake/
│   └── Code and supporting files for the transient cylinder-wake case
│
├── nasa_hump/
│   └── Code and supporting files for the statistically steady
│       NASA wall-mounted hump case
│
├── requirements.txt
├── verify_manuscript_config.py
├── CITATION.cff
├── LICENSE
└── README.md
