import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import indicators
import utils
import processor



root_dir = "./historical"
df1 = utils.read_from_file("./historical/ac")

df1 = indicators.heiken_ashi(df1)
df1 = indicators.ema(df1)

# create a new datafram
df2 = pd.DataFrame(df1[["date"]], index=df1.index)

df2 = df2.join(processor.ema_volume_diff(df1))

x = df2[['ema_diff']].as_matrix()
x = x[np.logical_not(np.isnan(x))]
x = x.reshape(-1, 1)
x
max_abs_scaler = preprocessing.MaxAbsScaler()
x_abs = max_abs_scaler.fit_transform(x)

x_abs
x.shape[0]
plt.plot(np.arange(0,x.shape[0]), x)
plt.plot(np.arange(0,x.shape[0]), x_abs)
plt.show()

# df.plot.line(y="ha_close")
# df.plot.line(y="ema_20")
# df.plot.line(y="close")
# plt.show()
# files = get_file_list(root_dir)
#
# for f in files:
#     df = read_from_file(join(root_dir, f))
#     ndf = df.as_matrix()
#     print(ndf)


# dd.plot.hist(alpha=0.5, bins=5)
# # plt.scatter(np.arange(1,31), stds)
# plt.show()
