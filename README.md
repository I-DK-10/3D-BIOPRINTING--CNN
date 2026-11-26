3D Bioprinting Extrusion Classification
Project Overview

This project focuses on classifying extrusion quality in 3D bioprinting images using Convolutional Neural Networks (CNNs).

The model classifies images into three categories:

Good extrusion
Under-extrusion
Over-extrusion
Dataset

Original dataset: Zenodo – Synthetic and Real Imagery of Extrusion from Bioprinter

A subset of 2,400 images was extracted for experimentation, combining real and synthetic images and both line and infill printing patterns.

Dataset distribution:

Training: 1,920 images
Testing: 480 images
Methodology
Image resizing: 128 × 128
Pixel normalization: 0–1
Model: Baseline CNN
Optimizer: Adam
Loss function: Sparse categorical cross-entropy
Batch size: 32
Maximum epochs: 20
Evaluation: Accuracy, precision, recall, F1-score and confusion matrix
Experimental Results

Training and validation accuracy, loss curves, confusion matrix and classification metrics are available in the results/ directory.

Repository Structure
notebooks/: Training notebook
src/: Python source code
results/: Experimental plots and metrics
models/: Trained model checkpoints
dataset_sample/: Extracted dataset subset, subject to dataset licence
Reproducibility

Install dependencies using:

pip install -r requirements.txt

Run the baseline CNN notebook to reproduce the experiment.

Dataset Citation

Original dataset: https://zenodo.org/records/13939763
