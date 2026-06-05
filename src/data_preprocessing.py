import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():

    df = pd.read_csv(
        "data/Wholesale customers data.csv"
    )

    return df


def preprocess_data(df):

    X = df.copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler