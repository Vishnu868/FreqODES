# FreqODES: Frequency-Domain Guided Online Domain Adaptation with Expert-in-the-Loop Active Learning

Course project for **22AIE437 Medical Image Processing** (B.Tech CSE (AI), Amrita School of Artificial Intelligence).

**Group A12:** Vishnu Vardhan (CB.SC.U4AIE23043) · Nithya Sai (CB.SC.U4AIE23021)

This repository does two things:

1. **Reproduces** [ODES: Online Domain Adaptation with Expert Guidance for Medical Image Segmentation](https://papers.miccai.org/miccai-2025/paper/5017_paper.pdf) (Islam et al., MICCAI 2025) on **CHAOS T1-DUAL in-phase → out-of-phase MRI**. It starts from the [official code](https://github.com/ShazidAraf/ODES) and fills in the parts that are missing there.
2. **Proposes FreqODES**. ODES chooses which slices the expert labels using batch-norm (BN) statistics, a score that needs network forward passes. FreqODES replaces that score with an interpretable **frequency-domain shift score** that needs none: **FFT** radial-band energy plus **Haar DWT** sub-band energy. The two scores can also be combined (fused). The final method, **FreqODES-U**, multiplies the frequency shift by the model's prediction uncertainty.

| | |
|---|---|
| Task | 2-D multi-class segmentation: liver, right kidney, left kidney, spleen |
| Data | CHAOS MR T1-DUAL, 20 patients, 647 slices. Source = in-phase, target = out-of-phase, streamed one patient at a time |
| Backbone | DeepLabv3: ResNet-18 encoder with ASPP |
| Budget | `b = 1%` of pixels in each selected slice; `K = 10 / 50 / 100%` of slices per patient |
| Metrics | Dice, Jaccard (IoU), adaptation time, paired Wilcoxon test over the 20 patients |

## Headline result: FreqODES-U (3 seeds, Tesla T4)

FreqODES-U scores each slice as **frequency shift × prediction uncertainty**. The table gives mean Dice ± std over seeds 1, 2 and 12345. Δ and p come from a paired Wilcoxon test against ODES over 60 patient-runs (20 patients × 3 seeds).

| Selection score | K = 10 % Dice | Δ vs ODES (p) | IoU | Online pass | K = 50 % Dice |
|---|---|---|---|---|---|
| ODES (BN-KL) | 78.31 ± 1.38 | – | 65.51 | 154 s | **79.40 ± 0.18** |
| v1: frequency (FFT+DWT) | 78.54 ± 0.13 | +0.22 (0.908) | 65.63 | 104 s | 79.04 ± 0.15 |
| v2: freq × organ area × uncertainty | 78.66 ± 0.26 | +0.35 (0.199) | 65.84 | 105 s | 78.91 ± 0.27 |
| Ablation: freq × organ area | 78.36 ± 0.28 | +0.04 (0.700) | 65.57 | 104 s | – |
| **FreqODES-U: freq × uncertainty** | **79.11 ± 0.17** | **+0.80 (0.004)** | **66.43** | **104 s** | 79.13 ± 0.33 |
| Random | 78.29 ± 0.57 | −0.02 (0.232) | 65.40 | 102 s | 79.10 ± 0.23 |

- **K = 10 % (low budget):** FreqODES-U **beats ODES** by +0.80 Dice (p = 0.004).
  - It wins on every seed: 78.93 / 79.16 / 79.26 against ODES's 76.80 / 78.67 / 79.48.
  - It is about 8× more stable across seeds and about 33 % faster.
- **K = 50 %:** ODES stays slightly better (79.40 vs 79.13, p = 0.018).
- **Honest note:** the planned v2 included an organ-area term, and FreqODES-U is its best ablation. We report every variant.

## Single-seed study (v1, seed 12345)

Mean Dice / Jaccard (%) on CHAOS T1-DUAL in-phase → out-of-phase, with b = 1 % of pixels annotated in each selected slice.

| Pruning score | K = 10 % Dice | K = 10 % IoU | K = 50 % Dice | Online pass (K = 10) |
|---|---|---|---|---|
| Source only (no adaptation) | 54.97 (paper 52.99) | 42.74 | – | 49 s |
| ODES, all slices (K = 100) | 79.52 (paper 78.61) | 66.96 | – | 117 s |
| **BN-KL (ODES, replicated)** | 78.83 (paper 76.80) | 66.24 | **79.57** (paper 78.56) | 167 s |
| FFT (FreqODES) | 78.02 | 65.03 | 79.24 | **107 s** |
| DWT (FreqODES) | 78.15 | 65.28 | 79.09 | 109 s |
| FFT+DWT (FreqODES) | 78.20 | 65.27 | 79.20 | 109 s |
| Fused, λ = 0.25 | **79.09** | 66.37 | – | ~170 s |
| Fused, λ = 0.5 | 78.40 | 65.45 | 79.50 | 169 s |
| Random | 78.75 | 66.04 | 79.52 | 107 s |

- **Replication:** adaptation lifts Dice from about 55 to 79.5, as in the paper (53 → 78.6).
- **Shift detector:** the frequency score separates in-phase from out-of-phase slices with ROC-AUC **0.994** (FFT), with no network forward passes. That makes the online pass about **35 % faster**.
- **Pure frequency pruning:** trails ODES by 0.6–0.8 Dice at K = 10 % and by 0.3–0.5 at K = 50 % (paired Wilcoxon, p = 0.005–0.09).
- **Fused score:** matches ODES. λ = 0.25 gives the best K = 10 % result (79.09 vs 78.83), but the difference is not significant (p = 0.75).
- **Takeaway:** a good *domain-shift* detector is not automatically a good *annotation-value* detector. The [report](report/main.pdf) analyses why.

The full tables, ablations and analysis are in [`results/results_tables.md`](results/results_tables.md), [`results/v2_summary.json`](results/v2_summary.json), the [slides](presentation/) and the [report](report/).

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
run_all.sh              19-run single-seed sweep (v1)
run_v2.sh               33-run 3-seed sweep (v2 / FreqODES-U)
analyze_v2.py           3-seed statistics, v2 figure and LaTeX numbers
make_results_html.py    interactive results page -> FreqODES_results.html (open in a browser)
freq_analysis.py        IP vs OOP spectra, separability AUC, synthetic-shift check
make_report.py          tables, LaTeX number macros, qualitative figure
plot_curves.py          training loss and per-patient Dice curves
colab_run.sh            one-shot pipeline for a Colab GPU (data -> PDFs)
results/                JSON results, logs, figures, tables
presentation/           Beamer slides (main.tex, refs.bib, figures/, main.pdf = compiled slides)
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

# 5. FreqODES-U / v2: 3 seeds, statistics, figure and LaTeX numbers
bash run_v2.sh
python analyze_v2.py results
```

`ACTIVE.PRUNE` takes one of these values:

- `BN`: original ODES
- `FFT`, `DWT` or `FREQ`: FreqODES (`FREQ` combines FFT and DWT)
- `FUSED`: λ·BN + (1−λ)·FREQ
- `FREQV2`, `FREQAREA`, `FREQENT`: v2 and its ablations; `FREQENT` (frequency × uncertainty) is **FreqODES-U**
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
