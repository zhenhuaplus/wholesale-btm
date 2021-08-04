import pandas as pd
import numpy as np
import os
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


# CAISO RT
folder = "/Users/zhenhua/Desktop/price_data/caiso_2019/DA_RT_LMPs/0096WD_7_N001/RT/"

folder_data = pd.DataFrame()
for file in os.listdir(folder):
    if "csv" in file:
        print(file)
        data = pd.read_csv(folder + file)
        folder_data = folder_data.append(data, ignore_index=True)
        folder_data = folder_data.sort_values(by=['INTERVALSTARTTIME_GMT', 'OPR_DT', 'OPR_HR']).reset_index(drop=True)
    else:
        pass

folder_data = folder_data[folder_data["OPR_INTERVAL"] == 1]
data = folder_data.pivot_table(index=['INTERVALSTARTTIME_GMT', 'OPR_DT', 'OPR_HR'], columns="LMP_TYPE",
                               values="MW").reset_index().rename_axis(None, axis=1)
data["Datetime (hb)"] = pd.to_datetime(data["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
data["RT Price ($/kWh)"] = data["LMP"] / 1000
data = data[["Datetime (hb)", "RT Price ($/kWh)"]]
data = data.set_index("Datetime (hb)")

data_da = pd.read_csv("/Users/zhenhua/Desktop/price_data/caiso_2019/DA_RT_LMPs/0096WD_7_N001/2019_processed_da.csv")
data_rt = pd.DataFrame()
data_rt["Datetime (hb)"] = pd.to_datetime(data_da["INTERVALSTARTTIME_GMT"]).dt.tz_convert('America/Los_Angeles')
data_rt = data_rt.set_index("Datetime (hb)")

data_rt = pd.concat([data_rt, data], axis=1, join_axes=[data_rt.index])
data_rt = data_rt.fillna(0)
data_rt.to_csv(folder + "2019_processed_rt.csv")
