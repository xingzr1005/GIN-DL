# Efficient Online Origin-Destination Prediction

This repository contains a PyTorch implementation of a knowledge-driven
origin-destination (OD) prediction framework based on a Gravity
Model-Inspired Network and LSTM.

The current code implements the static GIN-LSTM style pipeline:

1. Predict origin-level outflows from historical OD sequences with LSTM.
2. Estimate OD flow probabilities with a gravity-inspired neural network.
3. Multiply predicted outflows and OD probabilities to generate the next OD matrix.
4. Use a two-stage training strategy with key-frame pretraining and online fine-tuning.

Paper:

> Zeren Xing, Beiyu Song, Edward Chung, Ashish Bhaskar, Azusa Toriumi,
> Takashi Oguchi, and Keshuang Tang, "Efficient Online Origin-Destination
> Prediction via a Knowledge-Driven Deep Learning Framework."


## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The code is written for Python 3.9+ and PyTorch.



## Citation

If you use this code, please cite:

```bibtex
@article{xing2026efficient,
  title   = {Efficient Online Origin-Destination Prediction via a Knowledge-Driven Deep Learning Framework},
  author  = {Xing, Zeren and Song, Beiyu and Chung, Edward and Bhaskar, Ashish and Toriumi, Azusa and Oguchi, Takashi and Tang, Keshuang},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Early Access}
}
```
