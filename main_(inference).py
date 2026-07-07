import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import joblib, torch

from DBConnection import *
from data import *
from model import *
from config import *

def predict(model, device, item_id, seq_price, seq_volume, date_features):
    model.eval()

    # (1) item_id
    item_ids = torch.tensor([item_id], dtype=torch.long).to(device)

    # (2) sequence
    seq_price = np.log1p(seq_price)
    seq_volume = np.log1p(seq_volume)

    seq = np.stack([seq_price, seq_volume], axis=1)  # (L, 2)
    seq_input = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 2)

    # (3) length
    seq_lengths = torch.tensor([len(seq_price)], dtype=torch.long).to(device)

    # (4) numeric feature (month sin/cos)
    numeric_features = torch.tensor([date_features], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_log_price = model(item_ids, seq_input, seq_lengths, numeric_features)

    pred_price = torch.expm1(pred_log_price).item()

    return pred_price

if __name__ == '__main__':
    print(f"[SYSTEM][START] Start at {datetime.now()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model_files = sorted(MODELS_PATH.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError("No .pth model found in models folder.")
    model_path = model_files[-1]

    print(f"[MODEL][LOAD] START: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = PricePredictModel(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"[MODEL][LOAD] -> SUCCESS")
    
    le = joblib.load(ckpt["encoder_path"])
    meta = joblib.load(ckpt["meta_path"])
    id_to_name = dict(zip(meta["ITEM_ID"], meta["ITEM_NAME"]))
    print(f"[ENCODER][LOAD] {ckpt['encoder_path']} -> SUCCESS")
    print(f"[META][LOAD] {ckpt['meta_path']} -> SUCCESS")
    model.to(device)

    # Inference
    created_time = datetime.now() # .strftime("%Y-%m-%d")
    results = []

    # Query the latest 10 transactions for items used during training
    item_codes = le.classes_.tolist()
    
    if not item_codes:
        raise ValueError("No item codes found in the LabelEncoder.")
    placeholders = ", ".join(["?"] * len(item_codes))

    rows = select(
        table=f"""
        (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY ITEM_CODE
                    ORDER BY INOUT_DATE DESC
                ) AS RN
            FROM {PURCH_TABLE}
            WHERE ITEM_CODE IN ({placeholders})
        ) T
        """,
        columns="ITEM_CODE, ITEM_NAME, INOUT_DATE, INOUT_Q, INOUT_P",
        where="RN <= 10",
        params=tuple(item_codes),
        extra="ORDER BY ITEM_CODE, INOUT_DATE"
    )

    df_preprocessed = inf_preprocess(rows, le)
    latest_data_dt = max(df_preprocessed["date"])
    target_date = (latest_data_dt + relativedelta(months=1)).replace(day=1)

    loader = build_inf_dataloader(df_preprocessed, max_seq_len=10, batch_size=32)
    
    print(f"[INFERENCE][START]")
    with torch.no_grad():
        for item_ids, seq_input, seq_lengths, numeric_features in loader:
            
            item_ids = torch.as_tensor(item_ids, dtype=torch.long).to(device)
            seq_input = torch.as_tensor(seq_input, dtype=torch.float32).to(device)
            seq_lengths = torch.as_tensor(seq_lengths, dtype=torch.long).to(device)
            numeric_features = torch.as_tensor(numeric_features, dtype=torch.float32).to(device)
            
            preds = model(item_ids, seq_input, seq_lengths, numeric_features)
            preds = torch.expm1(preds).cpu().numpy()

            for i, p in enumerate(preds):
                results.append({
                    "item_id": item_ids[i].item(),
                    "item_name": id_to_name[item_ids[i].item()],
                    "pred_price": float(p)
                })
    print(f"[INFERENCE][END] total={len(results)}")
    
    # Insert into DB
    for r in results:
        insert_price_data = {
            "DATIME": created_time.strftime('%Y-%m-%d'),
            "LATEST_DATA_DT": str(latest_data_dt),
            "TARGET_DATE": str(target_date),
            "ITEM_NAME": r["item_name"],
            "ORDER_UNIT_P": None,
            "MONEY_UNIT": None,
            "ORDER_UNIT": None,
            "PRED_VALUE": r["pred_price"],
            "PRICE_DIFF_RATE": None,
            "REMARK": "",
        }
        insert(PRED_TABLE, insert_price_data)
        