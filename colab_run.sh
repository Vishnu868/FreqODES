#!/bin/bash
# Full FreqODES pipeline on a Colab GPU (run from the repo root):
#   data download -> preprocessing -> source training -> frequency analysis
#   -> 21-run experiment sweep -> tables/figures -> LaTeX slides and report PDFs
set -e
ROOT=$(pwd)
mkdir -p ../data_raw results/logs && cd ../data_raw
[ -d Train_Sets ] || { wget -q -O chaos.zip "https://zenodo.org/api/records/3431873/files/CHAOS_Train_Sets.zip/content"; unzip -q chaos.zip "Train_Sets/MR/*"; }
cd "$ROOT"
[ -d ../data/CHAOS ] || python prepare_chaos.py --chaos ../data_raw/Train_Sets/MR --out ../data/CHAOS
CK=checkpoint/CHAOS_T1_IP_OOP/CHAOS_T1_INPHASE.pt
[ -f $CK ] || python train_source.py --src ../data/CHAOS/T1DUAL_InPhase --out $CK --epochs 25 | tee results/logs/logs_train_source.txt
python freq_analysis.py && mkdir -p results/figures && cp results/freq_analysis.png results/figures/
# patients 3, 10 and 18 of the stream are saved for the qualitative figure
[ "${RERUN:-1}" == "1" ] && rm -rf results/json results/pred
SAVE_PRED=2,9,17 NAUG=20 NTHREADS=2 bash run_all.sh all
python make_report.py results results/figures
python plot_curves.py results results/figures
cp results/figures/results_tables.md results/figures/results_table1.csv results/
for d in presentation report; do mkdir -p $d/figures; cp results/figures/*.png results/figures/numbers.tex $d/figures/; done
if [ "${BUILD_PDF:-1}" == "1" ]; then
  command -v pdflatex >/dev/null || apt-get -qq install -y texlive-latex-extra texlive-publishers texlive-science texlive-bibtex-extra > /dev/null
  for d in presentation report; do
    (cd $d && pdflatex -interaction=nonstopmode main >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main > build.log; ls -la main.pdf)
  done
fi
