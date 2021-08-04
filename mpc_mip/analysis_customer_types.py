import pandas as pd
import numpy as np
import os

from mpc_mip.run import run_mpc

# Specify prices
raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_2019.csv")
tariff_price_data = pd.read_csv(
    "/Applications/storagevet2v101/StorageVET-master-git/Results_0425/output_run1_eCEF RS+DCM on/timeseries_results_runID1.csv")

# Specify load
doe_city_folder = "/Users/zhenhua/Desktop/price_data/DOE_C&I_load/C&I_Load_Profile_CA/"
doe_profile_folder_list = sorted([i for i in os.listdir(doe_city_folder) if "USA" in i])
system_load_min = 200

for doe_profile_folder in doe_profile_folder_list:

    for customer_path in sorted(os.listdir(doe_city_folder + doe_profile_folder)):
        doe_profile = pd.read_csv(doe_city_folder + doe_profile_folder + "/" + customer_path)

        for i in ["Restaurant", "Hospital", "Hotel", "Office", "School", "Market", "Apartment"]:
            if i in customer_path:
                customer_type = i
                location = doe_profile_folder.split("USA_")[1].split(".AP.")[0]
                break
            else:
                customer_type = None
                location = None

        if customer_type is not None:
            print("Runing MPC for {} in {}".format(customer_type, location))
            # Scale DOE profile load such that it has the same peak level as the system
            # normalization_ratio = np.mean(sorted(doe_profile["Electricity:Facility [kW](Hourly)"])[-10:]) / \
            #                       np.mean(sorted(raw_data["Site Load (kW)"])[-10:])
            doe_profile["Hour"] = doe_profile["Date/Time"].map(lambda x: x[-8:])
            doe_profile_daily_average = doe_profile.groupby("Hour")["Electricity:Facility [kW](Hourly)"].mean()
            normalization_ratio = min(doe_profile_daily_average) / system_load_min
            raw_data["Site Load (kW)"] = (doe_profile["Electricity:Facility [kW](Hourly)"]) / normalization_ratio

            # Read tariff data and specify energy prices
            raw_data["load_price"] = tariff_price_data["Energy Price ($/kWh)"]
            raw_data["charge_price"] = tariff_price_data["Energy Price ($/kWh)"]
            raw_data["discharge_price"] = tariff_price_data["Energy Price ($/kWh)"]

            # raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"]
            scenario_types = ["None", "Retail", "LMPs DA", "LMPs RT"]
            export_price_type = scenario_types[0]
            raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"] * 0 + 0
            # raw_data["export_price"] = tariff_price_data["Energy Price ($/kWh)"]

            # Specify params
            battery_size = 600
            battery_power = battery_size / 2
            demand_charge = 20
            one_way_efficiency = 0.92
            start_soc = 0.5
            max_soc = 1.0
            min_soc = 0.05
            n_horizon = 24
            n_control = 12

            results, results_dict = run_mpc(raw_data=raw_data,
                                            location=location, customer_type=customer_type, export_price_type=export_price_type,
                                            battery_size_kWh=battery_size, battery_power_kW=battery_power,
                                            min_soc=min_soc, max_soc=max_soc, one_way_efficiency=one_way_efficiency,
                                            demand_charge=demand_charge,
                                            start_soc=start_soc, n_horizon=n_horizon, n_control=n_control)
        else:
            pass
