# SZU-MNSR-Opt

Parametric OpenMC modeling and dummy-element layout optimization for the Shenzhen University Miniature Neutron Source Reactor (MNSR-SZ).

> **Project status:** Undergraduate research project.  
> This repository is currently private and contains unpublished research code.

## Overview

This project investigates the influence of dummy-element spatial arrangement on the neutronic performance of the Shenzhen University Miniature Neutron Source Reactor (MNSR-SZ).

A parametric reactor model is developed using **OpenMC**, and a **genetic algorithm (GA)** is coupled with Monte Carlo neutron transport calculations to search for improved spatial arrangements of five dummy elements while keeping the total fuel loading and major reactor parameters unchanged.

The primary optimization objective is radial fuel-pin power flattening. Additional quantities, including effective multiplication factor, reactivity, neutron-flux distribution, and central control-rod worth, are evaluated for selected candidate configurations.

The project focuses on relative neutronic comparison within a unified numerical model rather than high-precision prediction of absolute reactor safety parameters.

## Research Workflow

The overall computational workflow is:

```text
Dummy-element layout
        │
        ▼
Parameterized core geometry
        │
        ▼
OpenMC model generation
        │
        ▼
Monte Carlo criticality calculation
        │
        ├── keff
        ├── reactivity
        └── pin-wise kappa-fission tally
        │
        ▼
Power-distribution metrics
        │
        ├── radial power peaking factor
        └── coefficient of variation
        │
        ▼
Genetic-algorithm fitness evaluation
        │
        ▼
Selection / crossover / mutation
        │
        ▼
New candidate layouts
```

High-statistics independent OpenMC calculations are subsequently used to verify promising layouts and evaluate statistical uncertainty.

## Model Features

The current numerical model includes:

- Continuous-energy Monte Carlo neutron transport using OpenMC
- Parametric material definitions
- Concentric-ring core lattice representation
- 345 LEU UO2 fuel elements
- 5 dummy elements
- Central control-rod / guide-tube geometry
- Fixed structural positions
- Beryllium reflector
- Reactor-pool water region
- Parametric dummy-element positioning
- Pin-wise `kappa-fission` tallies
- Shannon-entropy monitoring for fission-source convergence
- Independent repeated calculations for statistical confirmation

The model is intended primarily for comparative neutronics studies. Some detailed engineering structures are simplified or omitted.

## Optimization Problem

Five dummy elements are selected from 350 ordinary lattice positions.

The number of possible spatial configurations is

\[
N = \binom{350}{5} = 42,530,162,570.
\]

Because exhaustive enumeration is computationally impractical when every candidate requires an OpenMC calculation, a genetic algorithm is used as a heuristic search method.

The primary optimization metric is the radial fuel-pin power peaking factor,

\[
F_r = \max_i \left(\frac{P_i}{\bar{P}}\right),
\]

where the pin-wise power proxy is obtained from the OpenMC `kappa-fission` tally.

The coefficient of variation of the normalized pin-power distribution is also used to characterize overall power uniformity.

## Repository Contents

### Core model

- `materials.py`  
  Defines the material compositions used in the OpenMC model.

- `geometry.py`  
  Builds the parameterized reactor geometry for a specified dummy-element arrangement and control-rod configuration.

### Optimization and statistical analysis

- `MNSR_formal_ga_workflow.py`  
  Main genetic-algorithm optimization workflow.

- `MNSR_statistical_confirmation.py`  
  Performs independent repeated high-statistics calculations for selected configurations.

- `MNSR_pooled_power_figures.py`  
  Pools repeated pin-power results and generates comparison figures.

### Control-rod analysis

- `MNSR_control_rod_position_pilot.py`  
  Pilot implementation and verification of the movable control-rod model.

- `MNSR_control_rod_formal_workflow.py`  
  Formal control-rod worth calculations.

- `MNSR_control_rod_flux_workflow.py`  
  Neutron-flux comparison workflow.

### Development notebooks

The `*.ipynb` files contain model-development, verification, testing, and exploratory calculations used during the research process.

They are retained to document the development history of the numerical model but are not necessarily required for the final automated workflows.

## Software Environment

The project has been developed primarily using:

- Python 3.13
- OpenMC 0.15.3-based development environment
- NumPy
- Pandas
- Matplotlib
- h5py
- Pillow
- IPython / ipykernel
- WSL / Ubuntu
- VS Code

The main nuclear data library used in the calculations is based on **ENDF/B-VII.1**.

Detailed dependency information is provided in `environment.yml` and `requirements.txt`.

## Installation

The recommended way to reproduce the Python/OpenMC environment is to use Conda.

Clone the repository:

```bash
git clone https://github.com/Ahzzzzzzz/SZU-MNSR-Opt.git
cd SZU-MNSR-Opt
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate szu-mnsr-opt
```

The current development environment is based on:

- Python 3.13
- OpenMC 0.15.3
- NumPy
- Pandas
- Matplotlib
- h5py
- Pillow
- IPython
- ipykernel

The original research environment used a development build of OpenMC derived from the 0.15.3 codebase. For reproducibility and portability, `environment.yml` specifies the stable `openmc=0.15.3` release rather than the local development build.

### Nuclear Data

OpenMC nuclear data are **not included** in this repository.

The calculations in this project use an ENDF/B-VII.1-based OpenMC cross-section library. Users must install a compatible OpenMC nuclear data library separately and configure the `OPENMC_CROSS_SECTIONS` environment variable.

For example:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

The exact path depends on the local installation and should not be hard-coded into the repository.

### Python-only Dependencies

The Python-level dependencies are also listed in:

```text
requirements.txt
```

They can be installed with:

```bash
pip install -r requirements.txt
```

However, this does **not** install OpenMC itself or the required nuclear data library. For full reproduction of the computational environment, `environment.yml` is the recommended setup method.

## Generated Files

OpenMC-generated files and large Monte Carlo output files are intentionally excluded from version control, including:

```text
statepoint.*.h5
summary.h5
materials.xml
geometry.xml
settings.xml
tallies.xml
build/
results/
```

These files can be regenerated from the source code when the required OpenMC environment and nuclear data library are available.

## Reproducibility

Random seeds and major Monte Carlo calculation parameters are explicitly controlled in the formal workflows to improve reproducibility.

Because Monte Carlo calculations are stochastic, individual runs may exhibit small statistical differences. Selected candidate configurations are therefore evaluated using independent repeated calculations before final comparison.

## Current Status

The current repository contains the working research code used for:

- reference-model evaluation;
- dummy-element layout optimization;
- statistical confirmation;
- pin-power distribution analysis;
- neutron-flux comparison;
- control-rod worth analysis.

The associated research manuscript is still under preparation/review.

Results in this repository should therefore be regarded as part of an ongoing research project rather than a finalized engineering design or reactor safety evaluation.

## Disclaimer

This repository is intended for academic research and numerical-method development.

The model contains geometric and physical simplifications and has not been validated for reactor operation, licensing, safety analysis, or engineering decision-making.

## License

No open-source license has currently been assigned.

All rights are reserved while the research project remains unpublished.
