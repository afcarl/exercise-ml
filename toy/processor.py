import pandas as pd
from collections import namedtuple

def ema_volume_diff(df):
    return df.join(pd.Series(df["volume"] - df["ema_volume"], name="ema_diff"))

# lower all ha

# range prices

# standardization

# HA (bring to zero, standardize), RSI (range: 1 to -1, 50 as 0), VolDiff (range: 1 to -1), price: range to 0 to 100


# get buy signals
def get_signals(df):
    # start at 2 cause we need at least 2 days prior to test
    for i in range(2, df.shape[0]):
        ac = df.loc[i-2:i,"ac"].values # ac 2 days ago to now
        if ac[0] < 0 and ac[1] < 0 and ac[0] < ac[1] and ac[1] < ac[2] and ac[2] > 0:
            df.loc[i,"signal"] = 1
        elif ac[0] > 0 and ac[1] > 0 and ac[0] > ac[1] and ac[1] > ac[2] and ac[2] < 0:
            df.loc[i,"signal"] = -1
        else:
            df.loc[i,"signal"] = 0
        i += 1
    df = df.fillna(0)
    return df
