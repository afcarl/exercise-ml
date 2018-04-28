import pandas as pd
from datetime import datetime

df = pd.read_csv("./quant/data.csv",
delimiter='\t', \
header=None, \
skipinitialspace=True, \
names=['date', 'close', 'open', 'high', 'low', 'volume', 'change'])


# process change
df.change = df.change.map(lambda x : float(x.strip('%'))/100)

# process volume
df.volume = df.volume.map(str.strip)
df.volume = (df.volume.replace(r'[KM]+$', '', regex=True).astype(float) * \
    df.volume.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))

# process date
df.date = df.date.map(str.strip)
df.date = pd.to_datetime(df.date, format='%b %d, %Y')
