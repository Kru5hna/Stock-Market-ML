import pandas as pd

def create_regression_features(df):
    df = df.copy()
    
    df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
    df['Next_Return'] = df.groupby('Symbol')['Daily_Return'].shift(-1)
    df['Open_Return'] = (df['Open'] - df['Prev Close']) / df['Prev Close']
    df['High_Ratio'] = df['High'] / df['Close']
    df['Low_Ratio'] = df['Low'] / df['Close']
    df['VWAP_Ratio'] = df['VWAP'] / df['Close']
    
    df = df.dropna()

    X = df[['Date', 'Daily_Return', 'Open_Return', 'High_Ratio', 'Low_Ratio', 'VWAP_Ratio', 'Volume']]
    y = df['Next_Return']

    return X, y


def create_classification_features(df):
    df = df.copy()

    df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
    df['Next_Return'] = df.groupby('Symbol')['Daily_Return'].shift(-1)
    df['Target'] = (df['Next_Return'] > 0).astype(int)

    df['MA5'] = df.groupby('Symbol')['Close'].rolling(5).mean().reset_index(0, drop=True)
    df['MA5_Ratio'] = df['Close'] / df['MA5']
    df['Open_Return'] = (df['Open'] - df['Prev Close']) / df['Prev Close']
    df['High_Ratio'] = df['High'] / df['Close']
    df['Low_Ratio'] = df['Low'] / df['Close']
    df['VWAP_Ratio'] = df['VWAP'] / df['Close']

    df = df.dropna()

    X = df[['Date', 'Daily_Return', 'MA5_Ratio', 'Open_Return', 'High_Ratio', 'Low_Ratio', 'VWAP_Ratio', 'Volume']]
    y = df['Target']

    return X, y