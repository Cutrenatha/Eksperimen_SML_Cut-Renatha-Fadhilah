import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "../credit_scoring_raw.csv"
OUTPUT_DIR = "./credit_scoring_preprocessing"
TARGET_COLUMN = "Risk"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path, index_col=0)  # fix: hapus Unnamed: 0
    logger.info(f"  Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Handling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)  # fix: no inplace
                logger.info(f"  Filled '{col}' with median={fill_val:.4f}")
            else:
                fill_val = df[col].mode()[0]
                df[col] = df[col].fillna(fill_val)  # fix: no inplace
                logger.info(f"  Filled '{col}' with mode='{fill_val}'")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Removed {removed} duplicate rows. Remaining: {len(df)}")
    return df


def encode_categorical(df: pd.DataFrame, target_col: str) -> tuple:
    logger.info("Encoding categorical features...")
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col == target_col:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info(f"  Encoded: {col}")
    return df, encoders


def encode_target(df: pd.DataFrame, target_col: str) -> tuple:
    logger.info(f"Encoding target column: '{target_col}'")
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    logger.info(f"  Classes: {list(le.classes_)}")
    return df, le


def remove_outliers_iqr(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    logger.info("Removing outliers using IQR...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    before = len(df)
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    logger.info(f"  Removed {before - len(df)} outlier rows. Remaining: {len(df)}")
    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler


def save_artifacts(scaler, encoders, target_encoder, output_dir: str):
    artifact_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(scaler,         os.path.join(artifact_dir, "scaler.pkl"))
    joblib.dump(encoders,       os.path.join(artifact_dir, "encoders.pkl"))
    joblib.dump(target_encoder, os.path.join(artifact_dir, "target_encoder.pkl"))
    logger.info(f"Artifacts saved to: {artifact_dir}")


def preprocess_pipeline(
    raw_path: str   = RAW_DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    target_col: str = TARGET_COLUMN,
    test_size: float   = TEST_SIZE,
    random_state: int  = RANDOM_STATE,
) -> tuple:
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(raw_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df, encoders       = encode_categorical(df, target_col)
    df, target_encoder = encode_target(df, target_col)
    df = remove_outliers_iqr(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    train_df = X_train_scaled.copy()
    train_df[target_col] = y_train.values

    test_df = X_test_scaled.copy()
    test_df[target_col] = y_test.values

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv( os.path.join(output_dir, "test.csv"),  index=False)
    logger.info(f"Saved train.csv and test.csv to: {output_dir}")

    save_artifacts(scaler, encoders, target_encoder, output_dir)
    logger.info("Preprocessing pipeline complete.")
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = preprocess_pipeline()
    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    print(f"\nTrain preview:\n{train_df.head(3)}")