#!/bin/bash

GPU_LIST=(0)

# Which AggZO variant to run (single merged trainer: src/dptrainer_aggzo.py)
#   dpzo -> plain DP-AggZO          (was src/dptrainer_aggzo_est.py)
#   sc   -> symmetric extension     (was src/dptrainer_aggzo_symmetry.py)
#   lc   -> localization correction (was src/dptrainer_aggzo_localization.py)
ZO_METHOD=${ZO_METHOD:-lc}

# random_c=$(awk -v min=1 -v max=256 'BEGIN{srand(); print min + rand()*(max-min)}')
# THRESH_LIST=($random_c)  
THRESH_LIST=(5) 

BASE_LR_LIST=(50e-5) # space-separated base lr values; effective lr = BASE_LR / DPZERO_THRESHOLD

SEED_LIST=(42 42 42 42 42 42) # 13 21 42 87 100
RANDOM_DIRECTION_SEED_LIST=(100 100 100 100 100 100) 

num_gpus=${#GPU_LIST[@]}
num_thr=${#THRESH_LIST[@]}

echo "Using GPUs: ${GPU_LIST[@]}"
echo "ZO method: ${ZO_METHOD}"
echo "Sweeping DPZERO_THRESHOLD: ${THRESH_LIST[@]}"
echo

idx=0

while [ $idx -lt $num_thr ]; do
  echo "=== New batch ==="

  for g in "${GPU_LIST[@]}"; do
    if [ $idx -ge $num_thr ]; then
      break
    fi

    thr=${THRESH_LIST[$idx]}
    sed=${SEED_LIST[$idx]}
    dir_sed=${RANDOM_DIRECTION_SEED_LIST[$idx]}

    echo ">>> Launch on GPU ${g} with DPZERO_THRESHOLD=${thr}, SEED=${sed} and DIRECTION_SEED=${dir_sed}"

    CUDA_VISIBLE_DEVICES=${g} \
    ZO_METHOD=${ZO_METHOD} \
    DPZERO_PRIVACY_EPS=6 \
    DP_SAMPLE_RATE=0.0625 \
    STEP=1000 \
    SEED=${sed} \
    NUM_DIRECTION=64 \
    RANDOM_DIRECTION_SEED=${dir_sed} \
    BASE_LRS="${BASE_LR_LIST[*]}" \
    DPZERO_THRESHOLD=${thr} \
    ESTIMATE_CLIP=False \
    TASK="sst-5" \
    bash examples/dpaggzo.sh &

    idx=$((idx + 1))
  done

  # wait for every job in this batch to finish before starting the next one
  wait
  echo "=== Batch finished ==="
  echo
done

echo ">>> All thresholds finished."