# Price_Forecasting_Model

A deep learning-based model for forecasting future unit prices using historical transaction records and item-level features.

## Overview

- Time-Series Forecasting: Predicts the next transaction unit price from historical price and transaction volume sequences.
- Item Embedding: Learns item-specific characteristics through trainable embedding vectors.
- Sequential Modeling: Utilizes LSTM to capture temporal patterns in transaction history.
- Feature Engineering: Incorporates cyclic date features (month and day of week) as additional inputs.
- Automated Training: Supports scheduled model retraining using the latest transaction data.
- Automated Inference: Generates future price predictions for all trained items using the latest transaction history.
- Database Integration: Retrieves transaction records from MSSQL and stores training metadata and prediction results.

## Features

- Automatic preprocessing of transaction data.
- Label encoding for item identifiers.
- Training and validation dataset generation using time-based splitting.
- Periodic retraining using the most recent transaction history.
- Batch inference for all trained items.
- Model, metadata, and encoder persistence.
- Logging support for both training and inference processes.
- Scheduled monthly model retraining and batch inference using Windows Task Scheduler.

## Model Architecture

- Item Embedding Layer
- LSTM Sequence Encoder
- Date Feature Encoder
- Fully Connected Prediction Head

### Input Features

- Item ID
- Historical transaction prices
- Historical transaction volumes
- Month (sin, cos)
- Day of week (sin, cos)

### Output

- Predicted next transaction unit price

## Project Structure

- `main_(train).py`: Model training entry point.
- `main_(inference).py`: Batch inference entry point.
- `model.py`: Neural network architecture.
- `dataset.py`: Dataset, preprocessing, and DataLoader utilities.
- `db.py`: MSSQL database operations.
- `config.py`: Project configuration and database settings.
- `models/`: Saved trained model weights.
- `encoder/`: Saved LabelEncoder objects.
- `meta/`: Saved metadata for trained items.
- `logs/Train/`: Training logs.
- `logs/Inference/`: Inference logs.

## Database

The model uses an MSSQL database as the primary data source.

Main tables include:

- Purchase transaction table
- Training information table
- Prediction result table

The training pipeline automatically retrieves transaction records, while the inference pipeline stores predicted prices back into the database.

## Training

The model is trained using:

- Recent transaction history (configurable period)
- Time-based train/validation split
- Adam optimizer
- Learning rate scheduler
- Early stopping

Only items with sufficient transaction history are included in training.

## Inference

During inference:

- Only items used during training are selected.
- The latest transaction records for each item are retrieved.
- The trained model predicts the next transaction unit price.
- Results are automatically stored in the prediction table.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages used in this project:

- torch
- pandas
- numpy
- scikit-learn
- pyodbc
- matplotlib

## Usage

Train the model:

```bash
python main_(train).py
```

Run batch inference:

```bash
python main_(inference).py
```

Or execute the provided batch scripts for automated scheduling.

## Workflow

The project is designed for automated monthly model retraining and batch inference.

1. Retrieve the latest **3 years** of transaction data from the MSSQL database.
2. Preprocess and clean the transaction records.
3. Retrain the price forecasting model.
4. Save the trained model, LabelEncoder, and metadata.
5. Load the latest trained model.
6. Retrieve the **latest 10 transactions** for each item used during training.
7. Predict the next transaction unit price for every trained item.
8. Store prediction results in the database.

## Automation

The project supports fully automated monthly execution using Windows Task Scheduler.

Two batch scripts are provided:

- `run_train.bat`
  - Retrains the model using the latest transaction data.
  - Saves training logs under `logs/Train/`.

- `run_inference.bat`
  - Loads the latest trained model.
  - Predicts future prices for all trained items.
  - Saves inference logs under `logs/Inference/`.

Typical production schedule:

- Monthly model retraining
- Monthly batch inference after training completes

## Notes

- The model predicts the next transaction unit price for each trained item.
- Training and inference logs are automatically saved.
- Only items included during training are used for inference to ensure consistency with the LabelEncoder.
- The project is designed for scheduled monthly retraining and batch inference in production environments.