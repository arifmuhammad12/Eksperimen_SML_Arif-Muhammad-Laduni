import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path):
  
  df = pd.read_csv(path)

  return df


def remove_unused_columns(df):

  if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

  return df


def convert_data_types(df):

  df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"]
  )

  return df


def check_missing_values(df):

  print("Missing Values:")
  print(df.isnull().sum())

  return df


def remove_duplicates(df):

  df.drop_duplicates(inplace=True)

  return df


def encode_target(df):

  df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
  })

  return df


def encode_categorical_features(df):

  categorical_cols = df.select_dtypes(
    include=["object"]
  ).columns

  le = LabelEncoder()

  for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

  return df


def scale_numerical_features(df):

  numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
  ]

  scaler = StandardScaler()

  df[numerical_cols] = scaler.fit_transform(
    df[numerical_cols]
  )

  return df


def split_dataset(df):

  X = df.drop("Churn", axis=1)
  y = df["Churn"]

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
  )

  return X_train, X_test, y_train, y_test


def save_preprocessed_data(df, output_path):
  
  os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
  )

  df.to_csv(
    output_path,
    index=False
  )

  print(f"Dataset berhasil disimpan di: {output_path}")


def preprocessing_pipeline(input_path, output_path):

  df = load_data(input_path)
  df = remove_unused_columns(df)
  df = convert_data_types(df)
  df = check_missing_values(df)
  df = remove_duplicates(df)
  df = encode_target(df)
  df = encode_categorical_features(df)
  df = scale_numerical_features(df)

  save_preprocessed_data(
    df,
    output_path
  )

  X_train, X_test, y_train, y_test = split_dataset(df)

  print("\nPreprocessing selesai!")
  print("Shape X_train:", X_train.shape)
  print("Shape X_test :", X_test.shape)

  return X_train, X_test, y_train, y_test


if __name__ == "__main__":

  INPUT_PATH = "../Telco-Customer-Churn.csv"

  OUTPUT_PATH = (
    "telco_preprocessing/"
    "telco_clean.csv"
  )

  preprocessing_pipeline(
    INPUT_PATH,
    OUTPUT_PATH
  )