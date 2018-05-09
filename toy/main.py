import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

import utils

utils.write_signals()

# df = utils.read_from_file("./historical/ac")

#
#
# tf = ac[<0]
# ac.iloc[:150].plot.bar()
# plt.show()
#
# df = indicators.heiken_ashi(df)
# df = indicators.ema_volume(df)
# df = indicators.relative_strength_index(df)
# df = df.fillna(0)
# df = processor.ema_volume_diff(df)
#
# # df = df[df.ema_diff != 0]
# dd = df.ema_volume.iloc[:50]




# Create a new dataframe
scaler = preprocessing.MaxAbsScaler()
df2 = pd.DataFrame(df[["date"]], index=df.index)



df2["rsi_norm"] = df.rsi.apply(lambda x : x - 0.5 if x != 0 else 0 )
df2 = df2.join(pd.Series(scaler.fit_transform(df[["ema_diff"]]).flatten(), name="ema_diff_norm"))



# x = df2[['ema_diff']].as_matrix()
# x = x[np.logical_not(np.isnan(x))]
# x = x.reshape(-1, 1)
#
# max_abs_scaler = preprocessing.MaxAbsScaler()
# x_abs = max_abs_scaler.fit_transform(x)
# x_abs
# df2 = df2.join(pd.Series(x_abs.flatten(), name="abs_ema_diff"))
# x_abs

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,x.shape[0]), x)
# plt.title("Unchanged")
# plt.subplot(2,1,2)
# plt.plot(np.arange(0,x.shape[0]), x_abs)
# plt.title("Abs")
# plt.show()

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
