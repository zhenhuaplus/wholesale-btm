import pandas as pd
import numpy as np
import os

from mpc_mip.run import run_mpc

# Specify prices
raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_2019.csv")
prices = pd.read_csv("/Users/zhenhua/Desktop/price_data/caiso_2019/DA_RT_LMPs/0096WD_7_N001/2019_processed.csv")
tariff_price_data = pd.read_csv("/Applications/storagevet2v101/StorageVET-master-git/Results_0425/output_run1_eCEF RS+DCM on/timeseries_results_runID1.csv")

# Specify load
doe_city_folder = "/Users/zhenhua/Desktop/price_data/DOE_C&I_load/C&I_Load_Profile_CA/"
doe_profile_folder = "USA_CA_San.Francisco.Intl.AP.724940_TMY3"
customer_path = "RefBldgLargeOfficeNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv"

doe_profile = pd.read_csv(doe_city_folder + doe_profile_folder + "/" + customer_path)
location = doe_profile_folder.split("USA_")[1].split(".AP.")[0]
customer_type = [i for i in ["Restaurant", "Hospital", "Hotel", "Office", "School", "Market", "Apartment"]
                 if i in customer_path][0]
doe_profile["Hour"] = doe_profile["Date/Time"].map(lambda x: x[-8:])
doe_profile_daily_average = doe_profile.groupby("Hour")["Electricity:Facility [kW](Hourly)"].mean()
# Customer daily average minimum equals to 200
normalization_ratio = min(doe_profile_daily_average) / 200
raw_data["Site Load (kW)"] = (doe_profile["Electricity:Facility [kW](Hourly)"]) / normalization_ratio

print("Runing MPC for {} in {}".format(customer_type, location))

# Read tariff data and specify energy prices
raw_data["load_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["charge_price"] = tariff_price_data["Energy Price ($/kWh)"]
raw_data["discharge_price"] = tariff_price_data["Energy Price ($/kWh)"]

scenario_types = ["None", "Retail", "LMPs DA", "LMPs RT"]
for export_price_type in scenario_types:
    if export_price_type == "None":
        raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"] * 0 + 0
    elif export_price_type == "Retail":
        raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"]
    elif export_price_type == "LMPs DA":
        raw_data["export_price"] = prices["DA Price ($/kWh)"]
    elif export_price_type == "LMPs RT":
        raw_data["export_price"] = prices["RT Price ($/kWh)"]

    battery_size = 1000
    for battery_size in [300, 600, 800, 1000]:
        demand_charge = 20
        battery_power = battery_size / 2
        one_way_efficiency = 0.92
        start_soc = 0.5
        max_soc = 1.0
        min_soc = 0.0
        n_horizon = 24
        n_control = 12

        results, results_dict = run_mpc(raw_data=raw_data,
                                        location=location, customer_type=customer_type, export_price_type=export_price_type,
                                        battery_size_kWh=battery_size, battery_power_kW=battery_power,
                                        min_soc=min_soc, max_soc=max_soc, one_way_efficiency=one_way_efficiency,
                                        demand_charge=demand_charge,
                                        start_soc=start_soc, n_horizon=n_horizon, n_control=n_control)
