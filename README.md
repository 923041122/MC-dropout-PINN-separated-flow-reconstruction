# MC-Dropout PINN for Physics-Constrained Separated-Flow Reconstruction

Code, supporting data, evaluation utilities, and reproducibility materials associated with the manuscript:

**“Physics-Constrained Flow Reconstruction and Uncertainty Assessment in Separated Flows Using Monte Carlo Dropout”**

This repository contains the implementations and supporting materials for two separated-flow reconstruction cases:

1. a transient cylinder-wake case; and
2. a statistically steady NASA wall-mounted hump case.

The repository accompanies the revised manuscript and is intended to make the reported model configurations, evaluation procedures, uncertainty analyses, and revision-related reproducibility checks transparent and inspectable.

---

## Repository ownership and manuscript association

This repository is maintained by **Linlin Zhu** (GitHub account: `923041122`) and contains code and reproducibility materials associated with the manuscript:

**“Physics-Constrained Flow Reconstruction and Uncertainty Assessment in Separated Flows Using Monte Carlo Dropout.”**

Linlin Zhu is identified in the manuscript contribution statement for **Software, Validation, Data Curation, and Visualization**.

For citation and authorship metadata, please see [`CITATION.cff`](CITATION.cff).

The repository is distributed under the [MIT License](LICENSE).

---

## Scope of the repository

The framework studied in the associated manuscript is a **reference-conditioned, physics-regularized flow-reconstruction framework**.

Monte Carlo dropout is used to obtain an approximate measure of **conditional epistemic uncertainty** by retaining dropout during repeated stochastic forward passes.

The terminology used here should not be interpreted as claiming:

- exact Bayesian posterior inference;
- a new Bayesian inference algorithm;
- a complete turbulence-resolving CFD solver;
- a full SST-RANS or LES turbulence closure; or
- a full Bayesian turbulence model.

The physics residuals are used as **soft physical regularization** during reconstruction.

For the cylinder-wake case, the reference velocity and pressure fields originate from a turbulence-model-based CFD calculation. The reconstruction residual uses a simplified incompressible momentum formulation and does not reproduce the complete SST turbulence closure, including the SST transport equations, eddy-viscosity model, or modeled Reynolds-stress terms.

For the NASA wall-mounted hump case, LES mean-field and wall-pressure information are used to condition and assess the reconstruction. The simplified physics residual should therefore be interpreted as a regularization constraint rather than as the complete LES mean-flow governing system.

---

## Repository structure

```text
.
├── cylinder_wake/
│   ├── bayesian_uncertainty_plot.py
│   ├── benchmark_config.py
│   ├── benchmark_evaluate.py
│   ├── benchmark_tools.py
│   ├── benchmark_train.py
│   ├── learning_schedule.py
│   ├── pinn_model.py
│   ├── plot_dimensionless.py
│   ├── read_data.py
│   ├── train_dropout_ablation.py
│   ├── train_uv_modify.py
│   ├── uncertainty_ablation.py
│   └── wake_probe_spectrum.py
│
├── nasa_hump/
│   ├── LES_cp_nasahump2009.dat
│   ├── LES_meanfield_nasahump2009_*.dat
│   ├── LES_statistics_profiles_nasahump2009_*.dat
│   ├── benchmark_tools.py
│   └── supporting NASA hump training, evaluation,
│       benchmark, and audit scripts
│
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── verify_manuscript_config.py
