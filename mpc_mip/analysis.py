import pandas as pd
from mpc_mip.run import run_mpc

# Specify prices
raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_pjm_2019.csv")
tariff_price_data = pd.read_csv("/Applications/storagevet2v101/StorageVET-master-git/Results_0425/output_run1_eCEF RS+DCM on/timeseries_results_runID1.csv")
raw_data["load_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["charge_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["discharge_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"]

# Specify params
battery_size = 8000
battery_power = 2000
one_way_efficiency = 0.92
start_soc = 0.5
max_soc = 1.0
min_soc = 0.05
n_horizon = 24
n_control = 24
demand_charge = 40

results, results_dict = run_mpc(raw_data=raw_data, battery_size_kWh=battery_size, battery_power_kW=battery_power,
                                min_soc=min_soc, max_soc=max_soc, one_way_efficiency=one_way_efficiency,
                                demand_charge=demand_charge,
                                start_soc=start_soc, n_horizon=n_horizon, n_control=n_control)