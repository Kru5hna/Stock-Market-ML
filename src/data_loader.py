import pandas as pd

def load_data(path="data/NIFTY50_all.csv"):
    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])

    df = df.dropna()

    df['Next_close'] = df.groupby('Symbol')['Close'].shift(-1)

    df = df.dropna()

    return df