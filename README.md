# ML-Guided Example Selection for LLM-Based Alzheimer's Disease Classification

This repository contains the code used for Alzheimer's disease classification using MRI-derived FreeSurfer features and large language models.

The study evaluates different in-context learning example-selection methods, including Random, Cosine similarity, Euclidean distance, Delta-kNN, and the proposed ML-guided method.

## Framework

![Framework](figures/framework.png)

## Repository Structure

```text
src/
├── Settings.py
├── 01_freesurfer_processing.py
├── 02_create_feature_vectors.py
├── 03_create_data_split.py
├── 04_train_ml_models.py
├── 05_select_examples.py
├── 06_run_llms.py
├── 07_reasoning_analysis.py
└── 08_evaluation.py
