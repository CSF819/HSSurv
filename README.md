# HSSurv

HSSurv is a multimodal survival analysis project for TCGA cohorts. This repository keeps a streamlined training entry, split files, model definition, and environment description for local reproduction.

## Repository Structure

```text
.
├── environment.yml
├── main.py
├── models/
│   └── HSSurv/
│       ├── network.py
│       ├── util.py
│       └── vit.py
├── splits/
│   └── 5foldcv/
│       ├── tcga_blca_new/
│       ├── tcga_brca_new/
│       ├── tcga_gbmlgg_new/
│       ├── tcga_luad_new/
│       └── tcga_ucec_new/
└── utils/
    ├── dataset_survival.py
    ├── engine.py
    ├── loss_factory_new.py
    ├── optimizer.py
    ├── options.py
    └── scheduler.py
```

## Environment

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate hssurv
```

## Data Preparation

This project expects pre-extracted pathology features and tabular omics data.

### Pathology Features

For each cohort, the feature root should follow this structure:

```text
DATA_ROOT/
├── slide/
│   ├── XXX.pt
│   └── ...
└── patch/
    ├── XXX.pt
    └── ...
```

- `slide/` stores slide-level features.
- `patch/` stores patch-level features.
- Feature file names should match `slide_id` values in the cohort CSV.

### Clinical / Omics CSV

`main.py` currently loads cohort CSV files from this local path pattern:

```text
./csv/tcga_{dataset}_all_clean_filtered.csv
```

Before running the project on another machine, update that path in [main.py] to match your local dataset location.

### Split Files

The training entry uses the following split directories by default:

- `splits/5foldcv/tcga_blca_new`
- `splits/5foldcv/tcga_brca_new`
- `splits/5foldcv/tcga_gbmlgg_new`
- `splits/5foldcv/tcga_luad_new`
- `splits/5foldcv/tcga_ucec_new`

Each directory should contain:

```text
splits_0.csv
splits_1.csv
splits_2.csv
splits_3.csv
splits_4.csv
```

## Training

The current training entry is:

```bash
python main.py HSSurv
```

A more explicit example is:

```bash
python main.py HSSurv --sets luad --fold 0,1,2,3,4 --which_splits 5foldcv
```

Useful arguments are defined in [utils/options.py], including:

- `--data_root_dir`
- `--sets`
- `--fold`
- `--num_epoch`
- `--lr`
- `--weight_decay`
- `--loss`

## Output

Training outputs are written under:

```text
./results/
```

For each dataset and fold, the project saves:

- per-run CSV summaries
- TensorBoard logs
- best model checkpoints
- run configuration snapshots

## Notes

- The codebase still contains a few machine-specific absolute paths and should be cleaned further before broader release.
- `main.py` is the active training entry point in this repository.
- The repository currently focuses on training code and split definitions rather than full raw-data preprocessing.
