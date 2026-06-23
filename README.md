# Hyperparameters Over Architecture

**A Controlled Comparison of Neural Networks for Aggregate Loss Reserving**

Qiheng Guo — Department of Mathematical Sciences, Ball State University

## Overview

This repository contains the code, model architectures, and experiment scripts for reproducing the results in:

> *Hyperparameters Over Architecture: A Controlled Comparison of Neural Networks for Aggregate Loss Reserving*
> Submitted to *Risks*, 2026.

We compare three neural network architectures for aggregate loss triangle reserving under identical data, training protocols, and evaluation procedures:

1. **GRU Baseline** — A faithful reimplementation of the [Kuo (2019) DeepTriangle](https://doi.org/10.3390/risks7030097)
2. **GRU + Attention (masked)** — Baseline augmented with single-head scaled dot-product attention and explicit padding masks
3. **GRU + Attention (unmasked)** — Ablation variant without padding masks

**Key findings:**
- The GRU Baseline wins. Both attention variants suffer from "attention collapse" — a bimodal failure mode where ~50% of training runs degenerate to naive mean predictions.
- Hyperparameter tuning (especially learning rate) yields larger accuracy gains than any architectural modification.
- Neural network advantages over Chain-Ladder concentrate on immature accident years.

## Repository Structure

```
├── replicate.py                 # One-command replication (verify or full CAS pipeline)
├── models.py                    # Architecture definitions (GRU Baseline, GRU+Attention masked/unmasked)
├── train.py                     # Training loop with early stopping, masked MSE loss
├── data_prep.py                 # DataManager: loading, normalization, temporal splitting
├── evaluate.py                  # MAPE/RMSPE computation, per-company evaluation
├── loss.py                      # Masked MSE loss function
├── benchmarks.py                # Mack Chain-Ladder, ODP Bootstrap, Bornhuetter-Ferguson
├── run_phase1.py                # Phase 1: Fixed-HP comparison (3 archs × 2 LOBs × 50 seeds)
├── run_phase2.py                # Phase 2: GRU Baseline HP screening (100 configs × 1 seed)
├── run_phase2_multiseed.py      # Phase 2: GRU Baseline multi-seed validation (top 20 × 5 seeds)
├── run_temporal.py              # Temporal robustness: rolling-origin cross-validation
├── run_kuo_comparison.py        # Kuo (2019) validation experiments
├── analyze_results.py           # Phase 1/2 analysis and figure generation
├── analyze_maturity.py          # Maturity-stratified analysis
├── analyze_hp_sensitivity.py    # Paper partial-dependence figure
├── compute_rf_permutation_importance.py  # RF importance robustness check
├── make_attention_collapse_diagnostic.py  # Reviewer 2 diagnostic figure
├── requirements.txt             # Python dependencies
├── data/
│   └── README.md                # Data description and sourcing instructions
├── paper/
│   ├── paper.tex                # Manuscript source
│   └── Definitions/             # MDPI journal class files
└── results/                     # Shipped pre-computed source results
    ├── diagnostics/             # De-identified diagnostic data and robustness-check JSON
    └── phase2/                  # WC and native PPA Phase 2 result folders
```

Note: generated figures are intentionally not shipped. `python replicate.py` recreates `results/figures/` from the pre-computed source results. The attention-collapse diagnostic uses `results/diagnostics/attention_collapse_diagnostic_data.json`, a de-identified plot-data file containing loss curves and anonymized actual-vs-predicted ultimate-ratio coordinates. The Random Forest robustness check writes `results/diagnostics/rf_permutation_importance_check.json` from the shipped Phase 2 summaries. A top-level `figures/` directory, if present in a local working copy, is stale scratch output.

## Data

The experiments use a proprietary S&P/NAIC Schedule P compilation covering 435 companies, accident years 1987–2019, and 10 development lags. The raw data CSVs are **not included** in this repository.

**To reproduce with public data:**

The [CAS Loss Reserving Data](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p) (Meyers & Shi, 2011; rehosted December 2025) provides a publicly available Schedule P dataset. Note the differences from our compilation:

| | Our Data | CAS Public Data |
|---|---|---|
| Accident years | 1987–2019 (33 yrs) | 1998–2007 (10 yrs) |
| Development lags | 0–9 | 1–10 |
| Companies | 435 | ~143 (PP Auto) |
| LOBs in scope | PP Auto, Workers' Comp | 6 LOBs (separate CSVs) |
| Group codes | S&P identifiers (C/P prefix) | Numeric NAIC codes |
| Columns | Includes loss ratios | Includes bulk/IBNR reserves |

To adapt the CAS data, you will need to:
1. Download the PP Auto and Workers' Comp CSVs from the link above
2. Rename columns to match the schema in `data/README.md` (e.g., `GRCODE` → `group_code`, `CumPaidLoss` → `cumulative_paid_loss`)
3. Compute incremental paid losses from cumulative values
4. Adjust development lag indexing (CAS uses 1–10; our code expects 0–9)

Place `triangle_sample.csv` and `triangle_company_info.csv` in the `data/` directory. See `data/README.md` for the full expected schema.

The data should contain:
- Accident years with 10 development lags (0–9)
- Workers' Compensation and Private Passenger Auto lines
- Fields: group code, accident year, development lag, incremental paid losses, case reserves, net earned premium

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.9+
- PyTorch 2.0+
- scikit-learn (Random Forest analysis)
- chainladder (classical benchmarks)
- pandas, numpy, matplotlib

## Quick Start: Replication

```bash
# Verify paper figures from pre-computed results (~30 seconds)
python replicate.py

# Full replication using CAS public data (~11 hours GPU)
python replicate.py --cas

# Run a specific step only
python replicate.py --cas --step phase1
python replicate.py --step diagnostic
python replicate.py --step rf-robustness

# See all options
python replicate.py --help
```

`replicate.py` orchestrates the entire pipeline. In **verify** mode (default), it regenerates all paper figures and tables from the shipped pre-computed results in `results/`. In **CAS** mode (`--cas`), it downloads the [CAS Loss Reserving Data](https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p), trains all models from scratch, and produces the full analysis. The `diagnostic` step is lightweight and regenerates the appendix attention-collapse diagnostic from de-identified plot data, so it does not require proprietary training data or private archive artifacts. The `rf-robustness` step is also lightweight and recomputes the permutation-importance check used to support the Random Forest importance caveat in the revised manuscript.

## Reproducing Results (Individual Scripts)

### Phase 1: Architecture Comparison (300 runs)

```bash
# All 3 architectures × 2 LOBs × 50 seeds
python run_phase1.py

# Or specific architectures/seeds
python run_phase1.py --archs gru_baseline --seeds 0 1 2 3 4
```

### Phase 2: Hyperparameter Sensitivity

```bash
# Stage 1: Screening (100 configs × 1 seed, GRU Baseline only)
# Stage 2: Validation (top 20 configs × 5 additional seeds)
python run_phase2_multiseed.py

# Or run stages separately
python run_phase2_multiseed.py --stage 1    # screening only
python run_phase2_multiseed.py --stage 2    # validation only
```

### Temporal Robustness

```bash
# Rolling-origin cross-validation (3 windows × 10 seeds)
python run_temporal.py
```

### Classical Benchmarks

```bash
# Mack Chain-Ladder, ODP Bootstrap, Bornhuetter-Ferguson
python benchmarks.py
```

### Analysis and Figures

```bash
python analyze_results.py           # Phase 1/2 figures and tables
python analyze_maturity.py          # Maturity-stratified analysis
python analyze_hp_sensitivity.py    # Partial-dependence figure
python replicate.py --step diagnostic  # Attention-collapse diagnostic figure
python replicate.py --step rf-robustness  # RF permutation-importance check
python replicate.py --step tables   # Print paper table/figure summary
```

## Computational Requirements

| Experiment | Runs | Est. Time (GPU) | Est. Time (CPU) |
|-----------|------|-----------------|-----------------|
| Phase 1 | 300 | ~6 hours | ~35 hours |
| Phase 2 screening | 100 | ~2 hours | ~16 hours |
| Phase 2 validation | 100 | ~2 hours | ~16 hours |
| Temporal robustness | 30 | ~1 hour | ~5 hours |
| **Total** | **~530** | **~11 hours** | **~72 hours** |

Experiments were conducted on an NVIDIA GB10 (DGX Spark, 128GB) and Apple M-series (Mac mini, 16GB).

## Citation

```bibtex
@article{guo2026architecture,
  title={Hyperparameters Over Architecture: A Controlled Comparison of Neural Networks for Aggregate Loss Reserving},
  author={Guo, Qiheng},
  journal={Risks},
  year={2026},
  note={Submitted}
}
```

## License

Code is released under the MIT License. The proprietary data is not redistributable.
