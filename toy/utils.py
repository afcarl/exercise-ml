import numpy as np
import pandas as pd
from os import walk
from os.path import join
import indicators
import utils
import processor
import requests
from datetime import datetime
from dateutil import tz

root_dir = "./historical"
target_dir = "./target"

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

    df = df[::-1]
    df = df.set_index(np.arange(0, df.shape[0]))
    return df

def write_signals():
    files = get_file_list(root_dir)

    for f in files:
        df = read_from_file(join(root_dir, f))
        df = indicators.ac(df)
        df = processor.get_signals(df)
        df = df[["date","signal"]]
        df[df.signal != 0].to_csv(join(target_dir, f))

def get_history(stock):
    url = "http://api.pse.tools/api/chart/history?symbol={0}&resolution=D&from=1495274769&to=1526378830".format(stock)
    response = requests.get(url).json()
    time_series  = pd.Series(response["t"], name="date")
    open_series  = pd.Series(response["o"], name="open")
    high_series = pd.Series(response["h"], name="high")
    low_series   = pd.Series(response["l"], name="low")
    close_series = pd.Series(response["c"], name="close")
    volume_series = pd.Series(response["v"], name="volume")

    result = pd.concat([time_series, open_series, high_series, low_series, close_series, volume_series], axis=1)
    result.date = result.date.apply(convertUTC)
    return result


def convertUTC(date):
    d = datetime.utcfromtimestamp(date)
    day = str(d.day)
    month = str(d.month)
    year = str(d.year)
    return "{0}-{1}-{2}".format(year, month.zfill(2), day.zfill(2))

    # response = requests.get("http://api.pse.tools/api/stocks").json()
    # data = response["data"]
    # df = pd.io.json.json_normalize(data).set_index("symbol")
    # df.loc["ALI"]
