#!/bin/bash
# functional / none / random x 3 seeds, sequential (each arm holds a ~15GB frozen table on the GPU).
cd "$(dirname "$0")/.."
# Activate your environment first (e.g. conda activate cpg-repr-benchmark).
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
for seed in 17 42 97; do
  for arm in functional none random; do
    [ -f outputs/$arm/seed_$seed/results.json ] && continue
    python scripts/train.py --arm $arm --seed $seed >> logs_train_${arm}_${seed}.txt 2>&1
  done
done
