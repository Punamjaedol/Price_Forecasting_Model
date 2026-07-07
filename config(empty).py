from pathlib import Path

# MSSQL database connection settings
DB_HOST = ""
DB_USER = ""
DB_PASSWORD = ""
DB_PORT = None
DB_NAME = ""

# Database table names
PURCH_TABLE = ""
TRAIN_INFO_TABLE = ""
PRED_TABLE = ""

# Directory for saving trained model files
MODELS_PATH = Path(__file__).resolve().parent / "models"

# Directory for saving label encoders
LABEL_ENCODER_PATH = Path(__file__).resolve().parent / "encoder"

# Directory for saving model metadata
META_PATH = Path(__file__).resolve().parent / "meta"