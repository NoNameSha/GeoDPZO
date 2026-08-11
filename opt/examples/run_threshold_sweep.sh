#!/bin/bash

# available GPUs
GPU_LIST=(2)   # adjust to the cards on your machine, e.g. (0 1) or (0 1 2 3)

# Which AggZO variant to run (single merged trainer: src/dptrainer_aggzo.py)
#   dpzo -> plain DP-AggZO          (was src/dp_trainer_aggzo.py)
#   sc   -> symmetric extension     (was src/dptrainer_aggzo_symmetry.py)
#   lc   -> localization correction (was src/dptrainer_aggzo_localization.py)
ZO_METHOD=${ZO_METHOD:-dpzo}
# Base learning rate numerator; effective lr = DP_BASE_LR / DP_CLIP
DP_BASE_LR=${DP_BASE_LR:-8e-4}

random_c=$(awk -v min=1 -v max=256 'BEGIN{srand(); print min + rand()*(max-min)}')

# DPZERO_THRESHOLD values to sweep
THRESH_LIST=(15) 
# THRESH_LIST=($random_c)  

SEED_LIST=(0 0 0)   #  1 2 3 4

num_gpus=${#GPU_LIST[@]}
num_thr=${#THRESH_LIST[@]}

echo "Using GPUs: ${GPU_LIST[@]}"
echo "ZO method: ${ZO_METHOD}"
echo "Sweeping DPZERO_THRESHOLD: ${THRESH_LIST[@]}"
echo

idx=0

# launch at most num_gpus jobs at a time, finish the batch before starting the next
while [ $idx -lt $num_thr ]; do
  echo "=== New batch ==="

  for g in "${GPU_LIST[@]}"; do
    # stop once the threshold list is exhausted
    if [ $idx -ge $num_thr ]; then
      break
    fi

    thr=${THRESH_LIST[$idx]}
    sed=${SEED_LIST[$idx]}
    echo ">>> Launch on GPU ${g} with DPZERO_THRESHOLD=${thr} and SEED=${sed}"

    CUDA_VISIBLE_DEVICES=${g} \
    ZO_METHOD=${ZO_METHOD} \
    DP_BASE_LR=${DP_BASE_LR} \
    MODEL=facebook/opt-350M \
    TASK=SST5 \
    MODE=ft \
    LR=6.09990013000600016065040888e-5 \
    SEED=${sed} \
    EPS=1e-3 \
    EVAL_STEPS=250 \
    DP_SAMPLE_RATE=0.0625 \
    DP_EPS=6 \
    STEPS=1000 \
    N=16 \
    DP_CLIP=${thr} \
    ESTIMATE_CLIP=False \
    bash examples/dpaggzo.sh &

    idx=$((idx + 1))
  done

  # wait for every job in this batch to finish before starting the next one
  wait
  echo "=== Batch finished ==="
  echo
done

echo ">>> All thresholds finished."
