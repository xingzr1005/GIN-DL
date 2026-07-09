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

## Repository Structure

```text
.
|-- Main.py                  # Command-line entry point
|-- Data_Container_OD.py     # Data loading, normalization, and DataLoader creation
|-- keyframe_extraction.py   # Key-frame extraction with MSSIM quantile thresholding
|-- GINDL.py                 # GIN-LSTM model
|-- Model_Trainer.py         # Pretraining, online training, and testing loops
|-- Metrics.py               # Evaluation metrics
|-- lib/                     # Legacy utility code
`-- requirements.txt         # Python dependencies
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The code is written for Python 3.9+ and PyTorch.



## Usage

### 1. Key-frame pretraining

```bash
python Main.py -mode pretrain --input_dir ../data
```


### 2. Online fine-tuning

```bash
python Main.py -mode train --input_dir ../data
```

### 3. Testing

```bash
python Main.py -mode test --input_dir ../data
```


## Key-Frame Extraction

Key frames are extracted by comparing consecutive OD matrices using MSSIM.
Given the MSSIM sequence, the code selects frames whose similarity is below
the specified quantile threshold and always keeps the first and last frames.

The default quantile is `0.4`:

```bash
python Main.py -mode pretrain --keyframe_quantile 0.4
```

Key-frame indices and MSSIM series are saved under the output directory.



## Citation

If you use this code, please cite:

```bibtex
@article{xing2026efficient,
  title   = {Efficient Online Origin-Destination Prediction via a Knowledge-Driven Deep Learning Framework},
  author  = {Xing, Zeren and Song, Beiyu and Chung, Edward and Bhaskar, Ashish and Toriumi, Azusa and Oguchi, Takashi and Tang, Keshuang},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Accepted for inclusion}
}
```
