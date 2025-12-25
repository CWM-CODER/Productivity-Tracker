import pandas as pd

def add_time_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour.fillna(0)
    return df

def encode_activity(df):
    df["activity_code"] = df["activity"].astype("category").cat.codes
    return df

def build_features(df):
    df = add_time_features(df)
    df = encode_activity(df)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    return df
