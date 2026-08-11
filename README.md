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
