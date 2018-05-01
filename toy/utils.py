import pandas as pd
from os import walk
from os.path import join

# read all files in a directory and get a list
def get_file_list(dir):
    filenames_only = []
    for (dirpath, dirnames, filenames) in walk(dir):
        filenames_only.extend(filenames)
        break
    return filenames_only

def read_from_file(file):
    df = pd.read_csv(file, \
    delimiter='\t', \
    header=None, \
    names=['date', 'close', 'open', 'high', 'low', 'volume', 'change'] )

    # strip all whitespace and commas
    df = df.astype(str)
    df = df.applymap(str.strip).applymap(lambda x: x.replace(',',''))

    # process change
    df.change = df.change.apply(lambda x : float(x.strip('%'))/100)
    # process volume
    # df.volume = df.volume.replace(r'[KM]+$', '', regex=True)
    # print(df.volume.to_string())

    df.volume = (df.volume.replace(r'[KM]+$', '', regex=True).astype(float) * \
        df.volume.str.extract(r'[\d\.]+([KM]+)', expand=False).\
            fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))

    # process date
    df.date = pd.to_datetime(df.date, format='%b %d %Y')

    # process OHLC
    df.close = df.close.astype(float)
    df.open = df.open.astype(float)
    df.high = df.high.astype(float)
    df.low = df.low.astype(float)

    # df = df.set_index("date")
    # df = df[::-1]
    return df
