#!/bin/bash

# Use a base random seed for reproducible but different random directions and DP noise across iterations:
# HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 DPZERO_PRIVACY_EPS=6 DP_SAMPLE_RATE=0.0625 STEP=1000 SEED=42 NUM_DIRECTION=64 RANDOM_DIRECTION_SEED=100 LR=4e-5 DPZERO_THRESHOLD=15 TASK="RTE" bash examples/dpaggzo.sh
#
# To sweep multiple base learning rates, pass a space-separated list:
#   BASE_LRS="1e-4 2e-4 5e-4" TASK="SST-2" bash examples/dpaggzo.sh
#
# To pick the AggZO variant (all three now live in src/dptrainer_aggzo.py):
#   ZO_METHOD=dpzo|sc|lc TASK="SST-2" bash examples/dpaggzo.sh

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-42}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-800}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-roberta-large}

# Which AggZO variant to run: dpzo (plain) | sc (symmetric extension) | lc (localization correction)
ZO_METHOD=${ZO_METHOD:-dpzo}

DPZERO_THRESHOLD=${DPZERO_THRESHOLD:-200.0}
ESTIMATE_CLIP=${ESTIMATE_CLIP:TRUE}
DPZERO_PRIVACY_EPS=${DPZERO_PRIVACY_EPS:-6.0} # -1 means non-DP, clipping threshold is then set to inf and no noise is injected
DPZERO_PRIVACY_DELTA=${DPZERO_PRIVACY_DELTA:-1e-5}
DP_SAMPLE_RATE=${DP_SAMPLE_RATE:-0.0416}
# 0.0416 for MNLI for expected BS=64
# 0.062 for SST-2 for expected BS=64
NUM_DIRECTION=${NUM_DIRECTION:-64} # number of random directions
RANDOM_DIRECTION_SEED=${RANDOM_DIRECTION_SEED:--1} # base seed for random directions and DP noise; -1 means use truly random seeds
SAMPLER_SEED=${SAMPLER_SEED:--1} # seed for batch sampling; -1 means use default seed

# Space-separated list of base learning rates to sweep; effective lr = BASE_LR / DPZERO_THRESHOLD
BASE_LRS=${BASE_LRS:-1e-4}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

# SST-2 sst-5 SNLI MNLI trec RTE

EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpzero-${TASK}-${EXTRA_TAG}}

for BASE_LR in $BASE_LRS; do
    # GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
    GR_TAG=seed$SEED-lr$LR-baselr$BASE_LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
    echo "Grid search tag: $GR_TAG"
    echo "Tag: $TAG"
    echo "Running with BASE_LR=$BASE_LR"

    TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
        bash examples/run_fewshot_aggzo.sh \
        --per_device_train_batch_size $BS \
        --learning_rate $LR \
        --eval_steps $EVAL_STEP \
        --weight_decay $WD \
        --zero_order_eps $EPS \
        --zero_order_optim \
        --lr_scheduler_type "constant" \
        --optimizer "sgd" \
        --efficient_zero_order \
        --dpzero \
        --zo_method $ZO_METHOD \
        --estimate_clip $ESTIMATE_CLIP\
        --dpzero_clip_threshold $DPZERO_THRESHOLD \
        --dpzero_base_lr $BASE_LR \
        --dp_epsilon $DPZERO_PRIVACY_EPS \
        --dp_delta $DPZERO_PRIVACY_DELTA \
        --dp_sample_rate $DP_SAMPLE_RATE \
        --n $NUM_DIRECTION \
        --random_direction_seed $RANDOM_DIRECTION_SEED \
        --sampler_seed $SAMPLER_SEED \
        $@
done