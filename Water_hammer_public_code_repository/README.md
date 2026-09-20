# Characteristic-constrained boundary learning for coupled water-hammer fields

This repository contains the authors' Python scripts and configuration files used for the numerical and machine-learning analyses reported in the manuscript **“Characteristic-constrained boundary learning for sparse reconstruction of coupled water-hammer fields.”**

The released code covers:

- one-dimensional four-equation fluid–structure interaction (FSSI) reference calculations;
- the characteristic-constrained PINN, a conventional coordinate PINN, and an ANN baseline;
- sparse-monitoring and parameter-input ablation studies;
- model comparison and error/peak-response post-processing;
- hydraulic-to-nominal-pipe-wall structural-response recovery;
- parameter screening and manuscript-figure generation; and
- the interface used for the independently obtained Xu et al. benchmark data.

## Repository layout

```text
.
├── PINN_FSSI_research_plan/
│   └── configs/                 # Registered JSON configurations
├── water16_reproduction/
│   ├── common/                  # Shared physics and PINN utilities
│   ├── tests/                   # Lightweight unit tests
│   ├── parametric_fssi_*.py     # Reference/model/training/evaluation scripts
│   ├── revised_*.py             # Comparison, ablation and screening scripts
│   └── *_figures.py             # Figure-generation/post-processing scripts
├── docs/
│   ├── DATA_AND_CODE_SCOPE.md
│   ├── THIRD_PARTY_MATERIALS.md
│   └── GITHUB_ZENODO_DOI_GUIDE_CN.md
├── CITATION.cff.template
├── LICENSE
├── requirements.txt
└── .gitignore
```

## Environment

The calculations were developed with Python and PyTorch on Linux. Create a clean environment and install the declared packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU execution, install the PyTorch build appropriate for the local CUDA version before installing the remaining requirements. No CUDA or hardware-specific path is hard-coded in this repository.

## Quick validation (no training)

From the repository root:

```bash
python -m compileall water16_reproduction
pytest -q water16_reproduction/tests
```

Some end-to-end tests and plotting scripts require previously generated model checkpoints or reference arrays. Those scripts report the missing path rather than downloading external material automatically.

## Main workflow

The commands below illustrate the intended sequence. Formal reference generation and training are computationally expensive; start with `smoke` or `audit` modes.

1. Generate/check the registered reference design:

   ```bash
   python -m water16_reproduction.parametric_fssi_reference \
     --config PINN_FSSI_research_plan/configs/parametric_fssi_forward_v1.json \
     --output-dir PINN_FSSI_research_plan/outputs/parametric_fssi_reference_v1 \
     --mode smoke --audit
   ```

2. Audit the characteristic-constrained model configuration without training:

   ```bash
   python -m water16_reproduction.parametric_fssi_hybrid \
     --config PINN_FSSI_research_plan/configs/parametric_fssi_hybrid_formal_v3_balanced_events.json \
     --output-dir PINN_FSSI_research_plan/outputs/parametric_fssi_hybrid_formal_v3_balanced_events \
     --mode audit
   ```

3. Train/evaluate the ANN and coordinate-PINN baselines with their registered configurations:

   ```bash
   python -m water16_reproduction.parametric_fssi_ann \
     --config PINN_FSSI_research_plan/configs/parametric_fssi_ann_baseline_v1.json \
     --output-dir PINN_FSSI_research_plan/outputs/ann_baseline --mode smoke

   python -m water16_reproduction.parametric_fssi_conventional_pinn \
     --config PINN_FSSI_research_plan/configs/parametric_fssi_conventional_pinn_baseline_v1.json \
     --output-dir PINN_FSSI_research_plan/outputs/coordinate_pinn_baseline --mode smoke
   ```

4. Run registered comparisons, ablations and post-processing only after the required result directories exist. Each program exposes its accepted paths through `--help`.

## Data and checkpoints

This code repository intentionally excludes large generated fields, model checkpoints, training logs and third-party raw data. This keeps the Git history stable and avoids redistributing material for which the authors do not control the licence. See [DATA_AND_CODE_SCOPE.md](docs/DATA_AND_CODE_SCOPE.md) and [THIRD_PARTY_MATERIALS.md](docs/THIRD_PARTY_MATERIALS.md).

## Citation and DOI

Before the first public release, copy `CITATION.cff.template` to `CITATION.cff` and replace every placeholder with the final author and DOI metadata. Then archive a tagged GitHub release through Zenodo. A Chinese step-by-step guide is provided in [GITHUB_ZENODO_DOI_GUIDE_CN.md](docs/GITHUB_ZENODO_DOI_GUIDE_CN.md).

## Licence

The authors' code in this repository is released under the MIT License. This licence does not cover third-party datasets, papers, repositories or software referenced by the scripts.
