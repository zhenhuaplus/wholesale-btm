import pandas as pd
import numpy as np
import os

from mpc_mip.run import run_mpc

# Specify prices
raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_2019.csv")
tariff_price_data = pd.read_csv("/Applications/storagevet2v101/StorageVET-master-git/Results_0425/output_run1_eCEF RS+DCM on/timeseries_results_runID1.csv")

customer_type = "System"
location = "CAISO"

# Read tariff data and specify energy prices
raw_data["load_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["charge_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["discharge_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["export_price"] = raw_data["DA Price ($/kWh)"] * 0
export_price_type = "None"

# Specify battery params
for battery_size in [400, 1000, 2000, 4000, 6000, 8000]:
    battery_size = battery_size
    battery_power = battery_size / 4
    one_way_efficiency = 0.92
    start_soc = 0.5
    max_soc = 1.0
    min_soc = 0.05
    n_horizon = 24
    n_control = 12
    demand_charge = 10

    results, results_dict = run_mpc(raw_data=raw_data,
                                    location=location, customer_type=customer_type, export_price_type=export_price_type,
                                    battery_size_kWh=battery_size, battery_power_kW=battery_power,
                                    min_soc=min_soc, max_soc=max_soc, one_way_efficiency=one_way_efficiency,
                                    demand_charge=demand_charge,
                                    start_soc=start_soc, n_horizon=n_horizon, n_control=n_control)
