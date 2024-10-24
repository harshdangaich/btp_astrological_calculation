# btp_astrological_calculations
# Astrological Marriage Prediction Model

This repository contains a neural network model implemented in PyTorch for predicting the success of a marriage based on astrological calculations. The model uses features such as house positions and planetary aspects to generate a binary prediction.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Input and Output](#input-and-output)
- [Training the Model](#training-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project is a machine learning model designed to predict the success of a marriage using astrological compatibility. It considers:
- The positions of 12 astrological houses for each partner.
- Planetary aspects (angular relationships between planets) to determine compatibility.

The model outputs a binary classification:
- `0`: Unsuccessful marriage.
- `1`: Successful marriage.

## Features
- **House Positions**: Represents the positions of the 12 astrological houses for both partners.
- **Planetary Aspects**: Represents the angular relationships between the planets.
- **Binary Classification**: The model predicts whether a marriage will be successful (`1`) or unsuccessful (`0`).

## Model Architecture
The model is a neural network implemented using PyTorch, with the following architecture:
- **Input Layer**: 34 features (24 for house positions + 10 for planetary aspects).
- **Hidden Layers**:
  - First hidden layer: 64 neurons, ReLU activation.
  - Second hidden layer: 32 neurons, ReLU activation.
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification).

The model uses binary cross-entropy loss for classification and a sigmoid function for final output.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/astrological-marriage-prediction.git
    cd astrological-marriage-prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have PyTorch installed. You can install it via:
    ```bash
    pip install torch torchvision
    ```

## Usage
1. Prepare your input data (normalized astrological house positions and planetary aspects).
2. Run the model with your dataset by executing:
    ```bash
    python train.py
    ```

### Example Input Format:
Input features include normalized house positions and planetary aspects:
```python
input_data = [
    # House positions for both partners (24 features)
    0.25, 0.48, 0.73, ..., 0.92,
    # Planetary aspects (10 features)
    1.0, -0.7, 0.6, ..., -0.8
]
