# btp_astrological_calculations
# Astrological Marriage Prediction Model

## Project Overview
This project uses a machine learning model to predict the success of a marriage based on astrological calculations. The model takes into account the positions of astrological houses and planetary aspects between two individuals to make a prediction.

## Inputs
- **Astrological House Positions**: 
  - 12 house positions for each partner (24 total features).
  - These values are normalized between 0 and 1 based on astrological principles.
  
- **Planetary Aspects**: 
  - 10 aspects indicating the angular relationships between planets for both partners.
  - These values are between -1 and 1, where the sign and magnitude represent compatibility.

## Outputs
- **Prediction**: 
  - A binary output indicating the likelihood of a successful marriage:
    - `0`: Unsuccessful
    - `1`: Successful

## Model Architecture
- Neural network built using PyTorch.
- 34 input features (24 house positions + 10 planetary aspects).
- Hidden layers with ReLU activations.
- Sigmoid activation for the final binary prediction.

## Objective
The goal of the model is to provide a predictive tool that uses astrological principles to forecast the success or failure of a marriage based on house positions and planetary aspects.

## Requirements
- Python 3.x
- PyTorch

## Usage
- Input astrological data for both individuals (house positions and planetary aspects).
- The model processes the data and returns a binary prediction for marriage success.

