from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class PriceSeqDataset(Dataset):
    def __init__(self, df, max_seq_len=10, min_history=1, date_range=None):
        """
        Generates training samples for time-series price prediction
        (historical price/volume sequence -> next price)
        """
        self.samples = []  # dict(item_id, seq_price, seq_volume, numeric_feat, target_price)
        self.max_seq_len = max_seq_len

        # Sort records by item and date
        df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

        # Generate sequences for each item
        for item_id, g in df.groupby("item_id"):
            g = g.reset_index(drop=True)
            prices = g["price"].values.astype(np.float32)
            volumes = g["volume"].values.astype(np.float32)
            dates = g["date"].values

            # Create one training sample for each time step
            for i in range(min_history, len(g)):
                cur_date = dates[i]

                # Filter samples by date range if specified
                if date_range is not None:
                    if not (np.datetime64(date_range[0]) <= cur_date < np.datetime64(date_range[1])):
                        continue

                start = max(0, i - max_seq_len)
                seq_price = prices[start:i]
                seq_volume = volumes[start:i]

                target_price = prices[i]

                # Generate date features
                numeric_feat = self._encode_date(cur_date)
                
                self.samples.append(
                    {
                        "item_id": item_id,
                        "seq_price": np.log1p(seq_price),  # Apply log scaling for numerical stability
                        "seq_volume": np.log1p(seq_volume),
                        "numeric_feat": numeric_feat,
                        "target_price": target_price,
                    }
                )

    @staticmethod
    def _encode_date(date):
        """Convert date into periodic features for model input"""
        date = np.datetime64(date, "D")
        dt = date.astype("datetime64[D]").item()
        month = dt.month
        dow = dt.weekday()
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        return np.array([month_sin, month_cos, dow_sin, dow_cos], dtype=np.float32)

    def __len__(self):
        """Return number of generated training samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return training sample corresponding to the index"""
        return self.samples[idx]

class PriceInferenceDataset(Dataset):
    def __init__(self, df, max_seq_len=10):
        """Prepare dataset for inference"""
        self.samples = []
        self.max_seq_len = max_seq_len

        # Sort records by item and date
        df = df.sort_values(["item_id", "date"])

        for item_id, g in df.groupby("item_id"):
            g = g.tail(max_seq_len)

            prices = np.log1p(g["price"].values)
            volumes = np.log1p(g["volume"].values)

            dates = g["date"].values
            last_date = dates[-1]

            numeric_feat = self._encode_date(last_date)

            self.samples.append({
                "item_id": item_id,
                "seq_price": prices,
                "seq_volume": volumes,
                "numeric_feat": numeric_feat,
                "last_date": last_date
            })
    
    @staticmethod
    def _encode_date(date):
        """날짜를 모델 입력용 주기성 특징으로 변환"""
        # 월, 요일을 주기성 특징으로 변환
        date = np.datetime64(date, "D")
        dt = date.astype("datetime64[D]").item()
        month = dt.month
        dow = dt.weekday()
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        return np.array([month_sin, month_cos, dow_sin, dow_cos], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    """Pads variable-length sequences to create batch data"""

    # Collect batch data
    item_ids = [b["item_id"] for b in batch]
    target_price = np.array([b["target_price"] for b in batch], dtype=np.float32)
    numeric_feat = np.stack([b["numeric_feat"] for b in batch])

    # Calculate sequence lengths
    seq_lengths = np.array([len(b["seq_price"]) for b in batch], dtype=np.int64)
    max_len = max(seq_lengths.max(), 1)

    # Padding processing
    seq_input = np.zeros((len(batch), max_len, 2), dtype=np.float32)

    for i, b in enumerate(batch):
        L = len(b["seq_price"])
        if L == 0:
            continue
        seq_input[i, :L, 0] = b["seq_price"]
        seq_input[i, :L, 1] = b["seq_volume"]

    seq_lengths = np.clip(seq_lengths, 1, None)
    
    return item_ids, seq_input, seq_lengths, numeric_feat, target_price

def inf_collate_fn(batch):
    """Pads variable-length sequences for inference batch data"""
    item_ids = [b["item_id"] for b in batch]

    seq_price = [b["seq_price"] for b in batch]
    seq_volume = [b["seq_volume"] for b in batch]

    max_len = max(len(x) for x in seq_price)

    seq_input = np.zeros((len(batch), max_len, 2), dtype=np.float32)
    seq_lengths = []

    for i in range(len(batch)):
        L = len(seq_price[i])
        seq_input[i, :L, 0] = seq_price[i]
        seq_input[i, :L, 1] = seq_volume[i]
        seq_lengths.append(L)

    numeric_features = np.stack([b["numeric_feat"] for b in batch])

    return item_ids, seq_input, seq_lengths, numeric_features

def preprocess(rows):
    """Preprocesses raw data into a DataFrame for training"""
    print("[DATA][PREPROCESS] prepare_dataset START")
    # Convert database query results to DataFrame
    df = pd.DataFrame.from_records(rows, columns=["ITEM_CODE", "ITEM_NAME", "INOUT_DATE", "INOUT_Q", "INOUT_P"])
    df = df.rename(columns={
        "ITEM_CODE": "item_code_raw",
        "INOUT_DATE": "date",
        "INOUT_Q": "volume",
        "INOUT_P": "price",
    })

    # 데이터 타입 변환
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    le = LabelEncoder()
    df["item_id"] = le.fit_transform(df["item_code_raw"])
    num_items = len(le.classes_)
    print(f"[DATA][ENCODE] LabelEncoder fitted | classes={num_items}")

    # Remove missing values
    df = df.dropna(subset=["price", "volume"])
    # Remove negative and zero transaction volumes
    df = df[df["volume"] > 0]
    # Remove abnormal prices
    df = df[
        ~(
            ((df["item_code_raw"] == "ITEM-A")   & (df["price"] == 10)) |
            ((df["item_code_raw"] == "ITEM-B") & (df["price"] == 12450)) |
            ((df["item_code_raw"] == "ITEM-C") & (df["price"] == 4080))
        )
    ].reset_index(drop=True)

    # Metadata for restoring item names
    meta = df[["item_id", "item_code_raw", "ITEM_NAME"]].drop_duplicates()
    meta = meta.rename(columns={
        "item_id": "ITEM_ID",
        "item_code_raw": "ITEM_CODE",
    })

    print(f"[DATA][CLEAN] after filtering | rows={len(df)}")
    print("[DATA][PREPROCESS] -> SUCCESS")
    return df, le, num_items, meta

def inf_preprocess(rows, le):
    """Preprocesses raw data into a DataFrame for inference"""
    df = pd.DataFrame.from_records(rows, columns=["ITEM_CODE", "ITEM_NAME", "INOUT_DATE", "INOUT_Q", "INOUT_P"])
    df = df.rename(columns={
        "ITEM_CODE": "item_code_raw",
        "INOUT_DATE": "date",
        "INOUT_Q": "volume",
        "INOUT_P": "price",
    })
    
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["item_id"] = le.transform(df["item_code_raw"])

    df = df.dropna(subset=["price", "volume"])
    df = df[df["volume"] > 0]

    return df

def build_dataloader(df, max_seq_len=10, batch_size=32, val_ratio=0.2):
    """Split training/validation data and create DataLoader"""
    print("[DATASET][DATALOADER] BUILD START")
    min_date, max_date = df["date"].min(), df["date"].max()
    cutoff_date = min_date + (max_date - min_date) * (1 - val_ratio)

    train_dataset = PriceSeqDataset(df, max_seq_len, date_range=(min_date, cutoff_date))
    val_dataset = PriceSeqDataset(df, max_seq_len, date_range=(cutoff_date, max_date + pd.Timedelta(days=1)))
    print("[DATASET][CREATE] train/val dataset created")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[DATASET][DATALOADER] train_batches={len(train_loader)} val_batches={len(val_loader)}")
    print("[DATASET][DATALOADER] SUCCESS")
    return train_loader, val_loader

def build_inf_dataloader(df, max_seq_len=10, batch_size=32):
    """Create DataLoader for inference"""
    print("[DATASET][DATALOADER] BUILD START")
    dataset = PriceInferenceDataset(df, max_seq_len)
    loader = DataLoader(dataset, batch_size=32, collate_fn=inf_collate_fn)
    print(f"[DATASET][DATALOADER] train_batches={len(loader)}")
    print("[DATASET][DATALOADER] SUCCESS")
    return loader

def eda(df):    
    # 1. Check negative and zero transaction volumes
    print("===== Transaction Volume Check =====")
    print(df["volume"].describe())

    print(f"\nNegative transaction volume count : {(df['volume'] < 0).sum()}")
    print(f"Zero transaction volume count    : {(df['volume'] == 0).sum()}")

    print("\nNegative transaction volume samples")
    print(df[df["volume"] < 0].head(20))

    # 2. Missing value check
    print("\n===== Missing value check =====")
    print(df.isnull().sum())

    # 3. Price distribution check
    print("\n===== Price statistics =====")
    print(df["price"].describe())

    print("\nTop 10 prices")
    print(df.nlargest(10, "price")[["item_code_raw", "date", "price"]])

    print("\nBottom 10 prices")
    print(df.nsmallest(10, "price")[["item_code_raw", "date", "price"]])

    # 4. Transaction volume distribution check
    print("\n===== Transaction volume statistics =====")
    print(df["volume"].describe())

    print("\nTop 10 transaction volumes")
    print(df.nlargest(10, "volume")[["item_code_raw", "date", "volume"]])

    # 5. Outlier detection (IQR method)
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_price = df[(df["price"] < lower) | (df["price"] > upper)]
    # ITEM_CODEs included in price outliers
    outlier_item_codes = outlier_price["item_code_raw"].unique()
    # Total data of the corresponding ITEM_CODE
    outlier_items_df = df[df["item_code_raw"].isin(outlier_item_codes)]

    print("\n===== Price outliers (IQR) =====")
    print(f"Outlier item codes : {list(outlier_item_codes)}")
    print(outlier_price.head(20))
    for item_code in list(outlier_item_codes):            
        item_df = df[df["item_code_raw"] == item_code]
        item_outlier = outlier_price[outlier_price["item_code_raw"] == item_code]

        print(f"\n===== {item_code} =====")
        print(f"Price minimum : {item_df['price'].min():,.0f}")
        print(f"Price maximum : {item_df['price'].max():,.0f}")
        print(f"Price median : {item_df['price'].median():,.0f}")
        print(f"Outlier ratio : {len(item_outlier) / len(item_df) * 100:.2f}%")
        print(f"Outlier count : {len(item_outlier)}")

    Q1 = df["volume"].quantile(0.25)
    Q3 = df["volume"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_volume = df[(df["volume"] < lower) | (df["volume"] > upper)]

    print("\n===== Transaction volume outliers (IQR) =====")
    print(f"Transaction volume minimum : {df['volume'].min():,.0f}")
    print(f"Transaction volume maximum : {df['volume'].max():,.0f}")
    print(f"Transaction volume median : {df['volume'].median():,.0f}")
    print(f"Transaction volume outlier ratio : {len(outlier_volume) / len(df) * 100:.2f}%")
    print(f"Transaction volume outlier count : {len(outlier_volume)}")

    print(outlier_volume.head(20))
