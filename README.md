# GeoDPZO
GeoDPZO: Symmetry-Aware Differentially Private Zeroth-Order Optimization for Efficient LLM Fine-Tuning

## Overview

GeoDPZO improves differentially private zeroth-order optimization by exploiting the implicit symmetry of multi-direction gradient estimators. We show that these estimators remain positively aligned with the expected zeroth-order gradient after clipping, which helps explain the stability gains of multi-direction aggregation.

Based on this insight, we develop two symmetry-aware clipping mechanisms, **GeoDPZO-SC** and **GeoDPZO-LC**, for different symmetry regimes. Compared with existing DPZO methods, GeoDPZO improves optimization stability and utility while reducing the need for costly additional perturbation directions.

Experiments across representative tasks and model families demonstrate that GeoDPZO enables more efficient private fine-tuning of large language models under the same privacy constraints. This artifact provides the code and experimental setup for validating these claims.

## Differential Privacy

We adopt the standard $(\epsilon,\delta)$-differential privacy definition.

A smaller privacy budget corresponds to a stronger privacy guarantee, generally at the cost of reduced model utility.

Throughout the experiments, we use

- $\delta = 10^{-5}$
- $\epsilon \in \{0.6, 3, 6\}$

Privacy accounting is implemented with the [Opacus](https://opacus.ai/) library based on the subsampled Gaussian mechanism.

## Environment Requirements

Our implementation is based on prior work [DP-AggZO](https://github.com/erguteb/dp-aggzo).

### Hardware

We recommend running the experiments on a Linux machine equipped with an NVIDIA RTX 4090 GPU with 24 GB of memory or better.

GPU memory requirements depend on the model size. Larger models, such as OPT-1.3B, require substantially more GPU memory than RoBERTa-large.

### Software Installation

The code has been tested with the following environment:

- Python 3.9.23
- PyTorch 2.4.0 + CUDA 12.1
- Transformers 4.28.1
- Opacus 1.4.0

The complete environment configuration is provided in `environments.yml`.

To create and activate the Conda environment, run:

```bash
conda env create -n geodpzo -f environments.yml
conda activate geodpzo
```

We highly recommend to test on RoBERTa-Large (355M), which takes less time.
In what follows, we will focus on RoBERTa-Large.

## Preparing the RoBERTa Experiments

Navigate to the `roberta` directory and prepare the datasets as follows.

The datasets can be downloaded from [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). If the primary server is unavailable, alternative download links are provided via [Dropbox](https://www.dropbox.com/scl/fi/s4tf8m0t40k85mhsybaru/datasets.tar?rlkey=55mq7r1s2lvs1l30ut08pbxe5&st=5yjtt4ba&dl=0) and [Google Drive](https://drive.google.com/file/d/1dq6MGyGyOdPTKLlhc1yEdr8tgjyuruxT/view?usp=sharing).

Please place the downloaded dataset archive in the `data/` directory, and then run:

```bash
cd data
bash prepare_datasets.sh && python sample_datasets.py
cd ..
```

## Functionality Test

For a quick functionality test, we recommend running **GeoDPZO-SC** on SST-2 with privacy budget `epsilon=6` and `K=16` zeroth-order directions.

In our implementation, the zeroth-order optimization method is specified by `ZO_METHOD`:

- `ZO_METHOD=sc`: GeoDPZO-SC
- `ZO_METHOD=lc`: GeoDPZO-LC
- `ZO_METHOD=dpzo`: DPZero/DPZO or DP-AggZO, depending on `NUM_DIRECTION`
  - `NUM_DIRECTION=1`: DPZero/vanilla DPZO
  - `NUM_DIRECTION>1`: DP-AggZO

To run GeoDPZO-SC on SST-2, use:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=16 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="75e-5" \
DPZERO_THRESHOLD=5 TASK="SST-2" \
ZO_METHOD=sc \
bash examples/dpaggzo.sh
```

Alternatively, you may manually edit `run_threshold_sweep.sh`, uncomment the command corresponding to the desired experiment, and run:

```bash
bash examples/run_threshold_sweep.sh
```


# Reproducibility

The following experiments are provided to reproduce the main empirical claims of GeoDPZO.

For artifact validation, we focus on RoBERTa-large, as experiments with larger OPT models require substantially more computation.


## C1-A: GeoDPZO-SC in the Small-\(K\) Regime

Claim **C1-A** evaluates whether **GeoDPZO-SC can achieve better utility than existing DPZO baselines in the small-\(K\) regime under the same privacy constraint**.

We compare GeoDPZO-SC against DP-AggZO and DPZero on SST-2 with privacy budget `epsilon=6`.


### C1-A-1: GeoDPZO-SC with \(K=16\)

Run GeoDPZO-SC by setting `ZO_METHOD=sc`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=16 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="75e-5" \
DPZERO_THRESHOLD=5 TASK="SST-2" \
ZO_METHOD=sc \
bash examples/dpaggzo.sh
```


### C1-A-2: DP-AggZO with \(K=16\)

To evaluate DP-AggZO under the same privacy budget and with the same number of zeroth-order directions, set `ZO_METHOD=dpzo` while keeping `NUM_DIRECTION=16`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=16 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="75e-5" \
DPZERO_THRESHOLD=5 TASK="SST-2" \
ZO_METHOD=dpzo \
bash examples/dpaggzo.sh
```

Here, `ZO_METHOD=dpzo` with `NUM_DIRECTION=16` corresponds to the multi-direction DP-AggZO baseline.


### Automatic Clipping Baseline

The automatic clipping variant can also be evaluated under the DP-AggZO setting.

To enable automatic clipping, navigate to the `src` directory and uncomment line 880 in:

```text
dptrainer_aggzo_est.py
```

Then run the corresponding DP-AggZO configuration with:

```bash
ZO_METHOD=dpzo
```


### C1-A-3: DPZero with \(K=1\)

DPZero, or vanilla DPZO, corresponds to the single-direction case. It can therefore be reproduced by setting `ZO_METHOD=dpzo` and `NUM_DIRECTION=1`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=1 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="75e-5" \
DPZERO_THRESHOLD=5 TASK="SST-2" \
ZO_METHOD=dpzo \
bash examples/dpaggzo.sh
```

Comparing **C1-A-1**, **C1-A-2**, and **C1-A-3** validates Claim **C1-A**, which examines whether GeoDPZO-SC provides improved utility over DP-AggZO and DPZero in the small-\(K\) regime under the same privacy constraint.


---

## C1-B: GeoDPZO-LC in the Large-\(K\) Regime

Claim **C1-B** evaluates whether **GeoDPZO-LC can achieve better utility than DP-AggZO in the large-\(K\) regime under the same privacy constraint**.

We perform the comparison on SST-5 with privacy budget `epsilon=6` and `K=64`.


### C1-B-1: GeoDPZO-LC with \(K=64\)

Run GeoDPZO-LC by setting `ZO_METHOD=lc`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="8e-5" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=lc \
bash examples/dpaggzo.sh
```


### C1-B-2: DP-AggZO with \(K=64\)

For comparison, run DP-AggZO with the same privacy budget, number of zeroth-order directions, and optimization configuration:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="8e-5" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=dpzo \
bash examples/dpaggzo.sh
```

The only method-level difference between the two experiments is therefore:

```text
ZO_METHOD=lc      # GeoDPZO-LC
ZO_METHOD=dpzo    # DP-AggZO
```

Comparing **C1-B-1** and **C1-B-2** validates Claim **C1-B**, which evaluates the utility advantage of GeoDPZO-LC over DP-AggZO in the large-\(K\) regime.


---

## C2: Utility-Efficiency Trade-off

Claim **C2** evaluates whether GeoDPZO can achieve a better trade-off between model utility and zeroth-order query cost.

We compare GeoDPZO using relatively small numbers of perturbation directions against DP-AggZO using larger values of \(K\).


### C2-1: GeoDPZO-SC with \(K=32\)

Run GeoDPZO-SC on SST-5 with `K=32`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=32 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="2e-4" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=sc \
bash examples/dpaggzo.sh
```


### C2-2: GeoDPZO-LC with \(K=64\)

Run GeoDPZO-LC with `K=64`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="2e-4" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=lc \
bash examples/dpaggzo.sh
```


### C2-3: DP-AggZO with \(K=64\)

For comparison, run DP-AggZO with `K=64`:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=64 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="2e-4" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=dpzo \
bash examples/dpaggzo.sh
```


### C2-4: DP-AggZO with \(K=256\)

To evaluate DP-AggZO with substantially more zeroth-order queries, increase `NUM_DIRECTION` to 256:

```bash
CUDA_VISIBLE_DEVICES=2 DPZERO_PRIVACY_EPS=6 \
DP_SAMPLE_RATE=0.0625 STEP=1000 \
SEED=42 NUM_DIRECTION=256 \
RANDOM_DIRECTION_SEED=100 \
BASE_LRS="2e-4" \
DPZERO_THRESHOLD=5 TASK="sst-5" \
ZO_METHOD=dpzo \
bash examples/dpaggzo.sh
```

The comparison can therefore be summarized as:

| Method | `ZO_METHOD` | \(K\) |
|---|---|---:|
| GeoDPZO-SC | `sc` | 32 |
| GeoDPZO-LC | `lc` | 64 |
| DP-AggZO | `dpzo` | 64 |
| DP-AggZO | `dpzo` | 256 |

Comparing these configurations validates Claim **C2**, which examines whether GeoDPZO can achieve improved utility with fewer zeroth-order perturbation directions and therefore a better utility-efficiency trade-off.


## Additional Notes

The zeroth-order optimization method is controlled by `ZO_METHOD`, while `NUM_DIRECTION` determines the number of random perturbation directions used by the optimizer.

In particular:

```text
ZO_METHOD=sc,   NUM_DIRECTION=K>1  -> GeoDPZO-SC
ZO_METHOD=lc,   NUM_DIRECTION=K>1  -> GeoDPZO-LC
ZO_METHOD=dpzo, NUM_DIRECTION=1    -> DPZero / vanilla DPZO
ZO_METHOD=dpzo, NUM_DIRECTION=K>1  -> DP-AggZO
```

Other experimental settings, including the privacy budget, sampling rate, number of optimization steps, learning rate, clipping threshold, and random seeds, can be controlled through the corresponding environment variables in the commands above.
