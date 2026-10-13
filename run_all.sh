#!/bin/bash
# Full experiment sweep for ODES vs FreqODES on CHAOS T1 IP -> OOP.
# Usage: bash run_all.sh <queue-id 1|2>   (two queues run in parallel, 4 threads each)
CFG=cfgs/CHAOS_T1_IP_OOP.yaml
COMMON="ACTIVE.N_AUG_PRUNE ${NAUG:-6}"
run() { name=$1; shift
  [ -f results/json/${name}_run_0.json ] && { echo "skip $name"; return; }
  echo "=== $name"; NTHREADS=${NTHREADS:-4} python demo_ODES.py --cfg $CFG $COMMON EXP_NAME $name "$@" > results/logs/$name.log 2>&1
  tail -1 results/logs/$name.log; }
mkdir -p results/logs
if [ "$1" == "1" ] || [ "$1" == "all" ]; then
  run source_only MODEL.Inference_Only True
  run ODES_K100 ACTIVE.BUDGET_SLICE 1.0
  for K in 0.1 0.5; do for P in FFT DWT FREQ RANDOM; do run ${P}_K${K} ACTIVE.PRUNE $P ACTIVE.BUDGET_SLICE $K; done; done
  run FREQ_K0.1_noDW ACTIVE.PRUNE FREQ ACTIVE.BUDGET_SLICE 0.1 ACTIVE.DIV_WEIGHT 0
  run RANDOM_K0.1_noDW ACTIVE.PRUNE RANDOM ACTIVE.BUDGET_SLICE 0.1 ACTIVE.DIV_WEIGHT 0
  run FUSED_K0.1_lam0.25 ACTIVE.PRUNE FUSED ACTIVE.BUDGET_SLICE 0.1 ACTIVE.FUSE_LAMBDA 0.25
  run FUSED_K0.1_lam0.75 ACTIVE.PRUNE FUSED ACTIVE.BUDGET_SLICE 0.1 ACTIVE.FUSE_LAMBDA 0.75
fi
if [ "$1" == "2" ] || [ "$1" == "all" ]; then
  for K in 0.1 0.5; do run BN_K${K} ACTIVE.PRUNE BN ACTIVE.BUDGET_SLICE $K; run FUSED_K${K}_lam0.5 ACTIVE.PRUNE FUSED ACTIVE.BUDGET_SLICE $K ACTIVE.FUSE_LAMBDA 0.5; done
  run BN_K0.1_noDW ACTIVE.PRUNE BN ACTIVE.BUDGET_SLICE 0.1 ACTIVE.DIV_WEIGHT 0
fi
echo QUEUE $1 DONE
