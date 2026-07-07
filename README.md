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

- `main_(train_model).py`: Model training entry point.
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

## Automated MLOps Workflow & Lifecycle

This project is built with a focus on long-term maintainability and automated model lifecycles, moving away from simple one-time script executions to a fully continuous pipeline.

1. **Data Ingestion**: Programmatically connects to the data warehouse to fetch historical time-series logs.
2. **Robust Preprocessing**: Automatically handles missing values, filters out noise/outliers (IQR method), and applies log-scaling for stability.
3. **Continuous Retraining**: Retrains the sequential model to capture evolving temporal patterns over time.
4. **Artifact Persistence**: Packages and exports the updated model weights, `LabelEncoder`, and item metadata simultaneously.
5. **Dynamic Inference Pipeline**: Loads the newly minted checkpoints and pulls the latest transaction context for all active items.
6. **Batch Forecasting**: Executes distributed predictions across all valid target classes.
7. **Storage Feedback Loop**: Updates the data store with future forecasts, closing the end-to-end automation loop.


##  Pipeline Automation & Orchestration

To achieve true zero-touch automation and demonstrate long-term system stability, the repository includes ready-to-use orchestration scripts. This architecture allows the entire system to run seamlessly via scheduling utilities (such as Windows Task Scheduler or Cron jobs).

Two integrated batch scripts orchestrate the pipeline:

* `run_train.bat`
  - Automates the full data ingestion and model retraining loop.
  - Automatically structures and isolates time-stamped training metrics under `logs/Train/`.

* `run_inference.bat`
  - Instantly hooks into the latest available model checkpoint.
  - Generates parallel future forecasts and records execution footprints under `logs/Inference/`.

### Key Architectural Strengths:
* **Decoupled Design**: Training and inference phases are structurally isolated, allowing independent scheduling intervals depending on data velocity.
* **Production-Ready Logging**: Every execution stream automatically pipes standard outputs and tracebacks into separate log files for reliable long-term monitoring and debugging.
* **End-to-End Autonomy**: Once configured, the system manages data flow, pipeline health, and prediction storage indefinitely without manual code modification.

## Notes

- The model predicts the next transaction unit price for each trained item.
- Training and inference logs are automatically saved.
- Only items included during training are used for inference to ensure consistency with the LabelEncoder.
- The project is designed for scheduled monthly retraining and batch inference in production environments.