import pandas as pd

def create_regression_features(df):

    X = df[['Prev Close','Open','High','Low','Volume','VWAP']]
    y = df['Next_close']

    return X, y


def create_classification_features(df):

    df['Target'] = (df['Next_close'] > df['Close']).astype(int)

    df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()

    df['MA5'] = df.groupby('Symbol')['Close'].rolling(5).mean().reset_index(0, drop=True)

    df = df.dropna()

    X = df[['Daily_Return','MA5','Volume','VWAP']]
    y = df['Target']

    return X, y