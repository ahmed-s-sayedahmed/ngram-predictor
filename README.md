# N-Gram Next Word Predictor

## Overview

This project implements a next-word prediction system using an N-gram language model. The model is trained on Sherlock Holmes novels by Arthur Conan Doyle and predicts the most probable next word based on a given input sequence.

The system uses Maximum Likelihood Estimation (MLE) with a backoff strategy: when a higher-order n-gram is not found, the model falls back to lower-order n-grams.

The project is structured into modular components: data preparation, model building, inference, and a command-line interface (CLI).

---

## Requirements

* Python 3.0 (recommended via Anaconda)
* Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ahmed-s-sayedahmed/ngram-predictor.git
cd ngram-predictor
```

### 2. Create and activate environment

```bash
conda create -n ngram python=3.10
conda activate ngram
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit `config/.env`:

```
TRAIN_RAW_DIR=data/raw/train/
TRAIN_TOKENS=data/processed/train_tokens.txt
MODEL=data/model/model.json
VOCAB=data/model/vocab.json
UNK_THRESHOLD=1
TOP_K=3
NGRAM_ORDER=4
```

### 5. Add dataset

Download Sherlock Holmes books from Project Gutenberg and place them in:

```
data/raw/train/
```

---

## Usage

### Run Data Preparation

```bash
python main.py --step dataprep
```

### Build Model

```bash
python main.py --step model
```

### Run Inference (CLI)

```bash
python main.py --step inference
```

### Run Full Pipeline

```bash
python main.py --step all
```

---

## Example

```
> holmes looked at
Predictions: ['the', 'him', 'watson']
```

---

## Project Structure

```
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/train/
│   ├── processed/
│   └── model/
├── src/
│   ├── data_prep/
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   ├── inference/
│   │   └── predictor.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
