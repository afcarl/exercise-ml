import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv("./quant/historical/ali.csv", \
delimiter='\t', \
header=None, \
names=['date', 'close', 'open', 'high', 'low', 'volume', 'change'] )

# strip all whitespace and commas
df = df.astype(str)
df = df.applymap(str.strip).applymap(lambda x: x.replace(',',''))

# process change
df.change = df.change.apply(lambda x : float(x.strip('%'))/100)

# process volume
df.volume = (df.volume.replace(r'[KM]+$', '', regex=True).astype(float) * \
    df.volume.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))

# process date
df.date = pd.to_datetime(df.date, format='%b %d %Y')

# process OHLC
df.close = df.close.astype(float)
df.open = df.open.astype(float)
df.high = df.high.astype(float)
df.low = df.low.astype(float)
#TODO fix BPI BDO
# histogram the changes
df.hist('change', bins=100)
std = df.change.std() * 100
std
# plt.show()
