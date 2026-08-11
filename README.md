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

## Installation

Our implementation builds upon the codebases of DPZero and MeZO.

The experiments were tested with the following software configuration:

- Python 3.9.23
- PyTorch 2.4.0 + CUDA 12.1
- Transformers 4.28.1
- Opacus 1.4.0

The complete Conda environment is provided in `environments.yml`.

Create and activate the environment using:

```bash
conda env create -n geodpzo -f environments.yml
conda activate geodpzo
