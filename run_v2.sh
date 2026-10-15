#!/bin/bash
# FreqODES v2 experiment: 3 seeds x {ODES (BN), FREQ (v1), FREQV2, RANDOM} x K in {10, 50} %,
# plus the two v2 ablations at K = 10 %. Design fixed before running; every run is reported.
CFG=cfgs/CHAOS_T1_IP_OOP.yaml
mkdir -p results/logs
run() { name=$1; shift
  [ -f results/json/${name}_run_0.json ] && { echo "skip $name"; return; }
  echo "=== $name"; python demo_ODES.py --cfg $CFG ACTIVE.N_AUG_PRUNE 20 EXP_NAME $name "$@" > results/logs/$name.log 2>&1
  tail -1 results/logs/$name.log; }
for S in 12345 1 2; do
  for K in 0.1 0.5; do
    for P in BN FREQ FREQV2 RANDOM; do run V2_${P}_K${K}_s${S} ACTIVE.PRUNE $P ACTIVE.BUDGET_SLICE $K RNG_SEED $S; done
  done
  for P in FREQAREA FREQENT; do run V2_${P}_K0.1_s${S} ACTIVE.PRUNE $P ACTIVE.BUDGET_SLICE 0.1 RNG_SEED $S; done
done
echo V2 DONE
