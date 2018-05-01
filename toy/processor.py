import pandas as pd

def ema_volume_diff(df):
    return pd.Series(df["volume"] - df["ema_volume"], name='ema_diff')

# lower all ha


# range prices

# standardization


# HA (bring to zero, standardize), RSI (range: 1 to -1, 50 as 0), VolDiff (range: 1 to -1), price: range to 0 to 100
