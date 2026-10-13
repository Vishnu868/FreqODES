# FreqODES: Frequency-Domain Guided Online Domain Adaptation with Expert-in-the-Loop Active Learning

Course project for **22AIE437 Medical Image Processing** (B.Tech CSE (AI), Amrita School of Artificial Intelligence).

This repository does two things:

1. **Reproduces** [ODES: Online Domain Adaptation with Expert Guidance for Medical Image Segmentation](https://papers.miccai.org/miccai-2025/paper/5017_paper.pdf) (Islam et al., MICCAI 2025) on **CHAOS T1-DUAL in-phase → out-of-phase MRI**. It starts from the [official code](https://github.com/ShazidAraf/ODES) and fills in the parts that are missing there.
2. **Proposes FreqODES**. ODES chooses which slices the expert labels using batch-norm (BN) statistics, a score that needs network forward passes. FreqODES replaces that score with an interpretable **frequency-domain shift score** that needs none: **FFT** radial-band energy plus **Haar DWT** sub-band energy. The two scores can also be combined (fused).

| | |
|---|---|
| Task | 2-D multi-class segmentation: liver, right kidney, left kidney, spleen |
| Data | CHAOS MR T1-DUAL, 20 patients, 647 slices. Source = in-phase, target = out-of-phase, streamed one patient at a time |
| Backbone | DeepLabv3: ResNet-18 encoder with ASPP |
| Budget | `b = 1%` of pixels in each selected slice; `K = 10 / 50 / 100%` of slices per patient |
| Metrics | Dice, Jaccard (IoU), adaptation time, paired Wilcoxon test over the 20 patients |

## Key results (Tesla T4, seed 1337)

| Pruning score | K = 10 % Dice | K = 50 % Dice | Online pass time (K = 10) |
|---|---|---|---|
| Source only (no adaptation) | 64.29 | – | – |
| ODES, all slices (K = 100) | 79.26 (paper 78.61) | – | – |
| **BN-KL (ODES, replicated)** | **78.38** (paper 76.80) | 79.26 | 176 s |
| FFT (FreqODES) | 77.31 | 79.34 | 116 s |
| FFT+DWT (FreqODES) | 77.65 | 78.91 | 117 s |
| Fused, λ = 0.5 | 78.15 | **79.35** | 178 s |
| Random | 78.15 | 79.21 | 115 s |

- **Shift detector:** the frequency score separates in-phase from out-of-phase slices with ROC-AUC **0.994** (FFT). Out-of-phase images carry more high-frequency energy.
- **K = 50 %:** FFT and fused pruning match annotating every slice, with half the expert effort.
- **K = 10 % (negative result):** the original BN score is still best. Pure frequency scores trail by about 0.7–1.1 Dice (p < 0.05). The fused score and random pruning are not significantly different from ODES.

The full tables, ablations and analysis are in [`results/results_tables.md`](results/results_tables.md), the [slides](presentation/) and the [report](report/).

## Repository layout

```
prepare_chaos.py        CHAOS DICOM/PNG -> per-patient (D,H,W) .tif stacks
train_source.py         source DeepLabv3 training (missing from upstream)
demo_ODES.py            online adaptation loop + Dice / Jaccard evaluation
methods/ODES.py         image pruning (BN / FFT / DWT / FREQ / FUSED / RANDOM), acquisition, BN update
methods/freq_shift.py   NEW: FFT + DWT frequency-domain domain-shift score
FFT_weight_2.py         farthest-first diversity weight map (imported upstream but never released; re-implemented from the paper)
model/                  DeepLabv3 (ResNet-18 + ASPP)
cfgs/CHAOS_T1_IP_OOP.yaml  experiment configuration
run_all.sh              21-run experiment sweep
freq_analysis.py        IP vs OOP spectra, separability AUC, synthetic-shift check
make_report.py          tables, LaTeX number macros, qualitative figure
plot_curves.py          training loss and per-patient Dice curves
colab_run.sh            one-shot pipeline for a Colab GPU (data -> PDFs)
results/                JSON results, logs, figures, tables
presentation/           Beamer slides (main.tex, refs.bib, figures/, main.pdf)
report/                 Elsevier CAS single-column report (main.tex, refs.bib, figures/, main.pdf)
```

## Changes to the upstream code

- Added a CHAOS preprocessing script and a source-training script. Upstream ships no checkpoint or training code.
- Re-implemented the missing `FFT_weight_2.py`: Algorithm 2 and the diversity map from the paper.
- Turned on the diversity weighting (Eq. 4). Upstream computed it but left the multiplication commented out. It can be switched off with `ACTIVE.DIV_WEIGHT 0`.
- Fixed a stale-variable bug where the acquisition map of a previous slice could replace the current one.
- Made the code device-agnostic (CUDA was hard-coded). Vectorised the BN KL distance; the values are unchanged.
- Added the config options `ACTIVE.PRUNE`, `ACTIVE.FUSE_LAMBDA`, `ACTIVE.DIV_WEIGHT`, `ACTIVE.FEATURE_DIVERSE_PRUNE`, `ACTIVE.LAMBDA_CONT` and `ACTIVE.N_AUG_PRUNE`.
- Added the Jaccard metric, per-patient results, JSON outputs and saved predictions.

## Reproduce

Requirements: Python 3.10+, PyTorch ≥ 2.0, plus:

```bash
pip install timm pydicom PyWavelets tifffile yacs iopath nibabel scikit-image opencv-python scipy scikit-learn matplotlib
```

**Everything in one go** (Google Colab with a T4 GPU, about 90 minutes): this downloads CHAOS, trains, runs all experiments, and builds the tables, figures and both PDFs.

```bash
bash colab_run.sh
```

**Step by step:**

```bash
# 1. data (CHAOS train set from Zenodo record 3431873)
wget -O chaos.zip "https://zenodo.org/api/records/3431873/files/CHAOS_Train_Sets.zip/content"
unzip -q chaos.zip "Train_Sets/MR/*" -d ../data_raw
python prepare_chaos.py --chaos ../data_raw/Train_Sets/MR --out ../data/CHAOS

# 2. source model
python train_source.py --src ../data/CHAOS/T1DUAL_InPhase --out checkpoint/CHAOS_T1_IP_OOP/CHAOS_T1_INPHASE.pt --epochs 25

# 3. one run, e.g. FreqODES (FFT+DWT) with 10 % of slices annotated
python demo_ODES.py --cfg cfgs/CHAOS_T1_IP_OOP.yaml ACTIVE.PRUNE FREQ ACTIVE.BUDGET_SLICE 0.1 EXP_NAME FREQ_K0.1

# 4. full sweep, tables and figures
NAUG=20 bash run_all.sh all
python freq_analysis.py
python make_report.py results results/figures
python plot_curves.py results results/figures
```

`ACTIVE.PRUNE` takes one of these values:

- `BN`: original ODES
- `FFT`, `DWT` or `FREQ`: FreqODES (`FREQ` combines FFT and DWT)
- `FUSED`: λ·BN + (1−λ)·FREQ
- `RANDOM`: random slice selection

## Slides and report

- `presentation/main.tex`: Beamer slides, 15 frames, following the 22AIE437 slide plan.
- `report/main.tex`: Elsevier CAS single-column journal report (`cas-sc`).

Every number in both documents is read from `figures/numbers.tex`, which `make_report.py` generates, so the documents always match the result files. To compile, upload the folder to Overleaf, or run `pdflatex main && bibtex main && pdflatex main && pdflatex main`. Fill in the group details in the `GROUP DETAILS` block at the top of each `main.tex`.

## Citation and license

This project builds on ODES. Please cite:

```
@inproceedings{islam2025odes,
  title={ODES: Domain Adaptation with Expert Guidance for Online Medical Image Segmentation},
  author={Md Shazid Islam and Sayak Nag and Arindam Dutta and Sk Miraj Ahmed and Fahim Faisal Niloy and Shreyangshu Bera and Amit K. Roy-Chowdhury},
  booktitle={MICCAI}, year={2025}
}
```

The ODES code is distributed under the UCR Academic Software License; see [`LICENSE_ODES.md`](LICENSE_ODES.md). Use is for educational and academic research only. *The Software used in this research was created by Amit Roy-Chowdhury and members of the Vision and Learning Group at UCR. © 2025 UCR.*

The CHAOS data is not redistributed here. Download it from [Zenodo](https://zenodo.org/records/3431873) under its own terms.
