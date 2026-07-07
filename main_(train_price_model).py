from Hydrotech.config import PURCH_TABLE
import copy, sys, time
from datetime import datetime, timedelta
import joblib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from DBConnection import *
from model import PricePredictModel
from data import *
from config import *

def loss_fn(pred_log_price, target_price, eps=1e-6):
    """Calculate Huber Loss in log scale"""
    target_log = torch.log1p(target_price.clamp(min=eps))
    return F.huber_loss(pred_log_price, target_log, delta=1.0)

@torch.no_grad()
def acc_fn(pred_log_price, target_price, tolerance=0.1, eps=1e-6):
    """Calculate ratio of predictions within tolerance and MAPE"""
    pred_price = torch.expm1(pred_log_price).clamp(min=0)
    ape = torch.abs(pred_price - target_price) / (target_price.abs() + eps)

    within_tol = (ape <= tolerance).float().mean()
    mape = ape.mean()
    return within_tol, mape

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        patience (int): Number of epochs to tolerate loss not improving
        min_delta (float): Minimum change considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # Best loss renewed
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0  # Reset counter
            print(f" -> [Best Model Saved] Lowest Loss reached: {self.best_loss:.4f}")
        else:
            # Loss not improving
            self.counter += 1
            print(f" -> [EarlyStopping Counter] {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        """Restore the weights of the best model at early stopping"""
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
            print(" -> [Restored] Loaded the best model weights.")

def run_epoch(model, loader, device, optimizer=None, tolerance=0.1):
    """
    Optimizer provided: train mode
    None: validation mode (torch.no_grad)
    Returns: (avg_loss, avg_acc)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
 
    with context:
        for item_ids, seq_input, seq_lengths, numeric_feat, target_price in loader:
            item_ids = torch.as_tensor(item_ids, dtype=torch.long).to(device)
            seq_input = torch.as_tensor(seq_input, dtype=torch.float32).to(device)
            seq_lengths = torch.as_tensor(seq_lengths, dtype=torch.long).to(device)
            numeric_feat = torch.as_tensor(numeric_feat, dtype=torch.float32).to(device)
            target_price = torch.as_tensor(target_price, dtype=torch.float32).to(device)
            
            pred_log_price = model(item_ids, seq_input, seq_lengths, numeric_feat)
            loss = loss_fn(pred_log_price, target_price)            
                        
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            within_tol, _ = acc_fn(pred_log_price, target_price, tolerance=tolerance)
 
            total_loss += loss.item()
            total_acc += within_tol.item()
            n_batches += 1
 
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=100, batch_size=4):
    # Initialize training information
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    print(f"[TRAIN][START] Device: {device}")
    train_start = time.time()
    model.train()
    
    # Train for each epoch
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)

        # Update Learning Rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Store Training History
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)

        # Print Training Results
        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[TRAIN][EARLY_STOPPING] {epoch+1} epoch.")
            break
 
    early_stopping.load_best_model(model)
    training_time = str(timedelta(seconds=int(time.time() - train_start)))
    print(f"[TRAIN][SUCCESS] training_time={training_time}")
    return model, history, training_time

def plot_training_history(history):
    """Visualize Loss and Accuracy trends based on training history"""
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss (log-scale Huber)")
    line1 = ax1.plot(epochs, history["loss"], color=colors[0], marker="o", label="Loss")
    line2 = ax1.plot(epochs, history["val_loss"], color=colors[1], marker="o", linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Shared Y-axis on the right: Accuracy metrics (0% ~ 100% increasing trend)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (within tolerance, %)")
    line3 = ax2.plot(epochs, history["acc"], color=colors[2], marker="o", linestyle="--", label="Accuracy")
    line4 = ax2.plot(epochs, history["val_acc"], color=colors[3], marker="o", linestyle="--", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor="black") 
    ax2.set_ylim(0, 100)  # Fixed range 0~100 since it is a percentage metric

    # Legend Integration
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("Price Prediction Model Training Metrics over Epochs")
    fig.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    print(f"[SYSTEM][START]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIN_HISTORY = 20
    YEARS = 3
    
    # Query transaction records from the last 3 years for items with at least 20 transactions
    rows = select(
        table=PURCH_TABLE,
        columns="ITEM_CODE, ITEM_NAME, INOUT_DATE, INOUT_Q, INOUT_P",
        where=f"""
            INOUT_DATE >= DATEADD(YEAR, -{YEARS}, GETDATE())
            AND ITEM_CODE IN (
                SELECT ITEM_CODE
                FROM {PURCH_TABLE}
                WHERE INOUT_DATE >= DATEADD(YEAR, -{YEARS}, GETDATE())
                GROUP BY ITEM_CODE
                HAVING COUNT(*) >= {MIN_HISTORY}
            )
        """
    )
        
    # Preprocess Data
    # eda(df_copy)
    
    df_preprocessed, le, num_items, meta = preprocess(rows)

    # Create DataLoader
    train_loader, val_loader = build_dataloader(df_preprocessed, max_seq_len=10, batch_size=32, val_ratio=0.2)
    
    # Model & Training Settings
    model = PricePredictModel(num_items=num_items).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
   
    # Train Model
    created_time = datetime.now()
    trained_model, history, training_time = train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=100)
    plot_training_history(history)
    
    # Save Trained Model, Label Encoder, and Metadata
    save_model_dir = MODELS_PATH
    save_le_dir = LABEL_ENCODER_PATH
    save_meta_dir = META_PATH

    save_model_dir.mkdir(parents=True, exist_ok=True)
    save_le_dir.mkdir(parents=True, exist_ok=True)
    save_meta_dir.mkdir(parents=True, exist_ok=True)
    model_name = 'train_' + created_time.strftime('%Y%m%d%H%M%S')
    model_path = save_model_dir / f"{model_name}.pth"    
    le_name = 'item_encoder__' + created_time.strftime('%Y%m%d%H%M%S')
    le_path = save_le_dir / f"{le_name}.pkl"
    meta_name = 'item_meta__' + created_time.strftime('%Y%m%d%H%M%S')
    meta_path = save_meta_dir / f"{meta_name}.pkl"

    checkpoint = {
        "config": {
            "num_items": num_items,
            "item_emb_dim": 16,
            "seq_hidden_dim": 32,
            "numeric_dim": 4,
            "hidden_dims": (64, 32),
            "dropout": 0.1,
            "rnn_layers": 1,
        },
        "state_dict": model.state_dict(),
        "encoder_path": str(le_path),
        "meta_path": str(meta_path)
    }
    
    torch.save(checkpoint, model_path)
    print(f"[MODEL][SAVE] {model_path} -> SUCCESS")
    joblib.dump(le, le_path)
    joblib.dump(meta, meta_path)
    print("[ENCODER][SAVE] SUCCESS")
    print("[META][SAVE] SUCCESS")

    # Save Training Information
    train_info = {
                    "MODEL_ALGORITHM": created_time.strftime('%Y-%m-%d'),
                    "MODEL_NAME": model_name,
                    "DATIME": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "LATEST_DATA_DT": str(df_preprocessed["date"].max()),
                    "TRAIN_TIME": training_time,
                    "LOSS": history["loss"][-1],
                    "VAL_LOSS": history["val_acc"][-1],
                    "LOSS_TYPE": "log-huber",
                    "THRESHOLD": 0.1,
                    "ACCURACY": history["val_acc"][-1],
                    "REMARK": "",
                }

    try:
        insert(TRAIN_INFO_TABLE, train_info)
        print("[SYSTEM][SAVE] Training Information -> SUCCESS")
    except Exception as e:
        print(f"[SYSTEM][ERROR] Save Training Information -> {e}")
        sys.exit(1)
    