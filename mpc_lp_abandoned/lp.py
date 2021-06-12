import pandas as pd
import numpy as np
from datetime import timedelta
import cvxpy as cp
import json


def run_optimization(daily_load, energy_price,
                     battery_size_kWh, battery_power_kW,
                     min_soc, max_soc, current_soc, one_way_efficiency):
    # Obtain parameters
    battery_energy_min = battery_size_kWh * min_soc
    battery_energy_max = battery_size_kWh * max_soc

    # Design optimization-based control policies
    # Define variables
    load = daily_load["Site Load (kW)"]
    net_load_after_storage = cp.Variable(daily_load.shape[0])
    battery_power = cp.Variable(daily_load.shape[0])
    battery_energy = cp.Variable(daily_load.shape[0] + 1)

    battery_discharge_to_load = cp.minimum(cp.maximum(battery_power, 0), load)
    battery_discharge_to_grid = cp.maximum(cp.maximum(battery_power, 0) - load, 0)

    # Initialize constraints
    constraints = []

    # General battery power/energy constraints
    constraints += [-battery_power_kW <= battery_power, battery_power <= battery_power_kW]
    constraints += [battery_energy_min <= battery_energy, battery_energy <= battery_energy_max]

    # General load/net load constraints

    # constraints += [net_load_after_storage == np.array(load) - battery_charge - battery_discharge_to_load]
    constraints += [net_load_after_storage == np.array(load) - battery_power]

    # Export constraints
    export_to_grid_allowed = True
    if export_to_grid_allowed:
        pass
    else:
        constraints += [battery_discharge <= load]

    # Battery energy dynamics
    constraints += [battery_energy[0] == battery_size_kWh * current_soc]
    for i in range(daily_load.shape[0]):
        # constraints += [battery_energy[i + 1] == battery_energy[i] -
        #                 battery_charge[i] * one_way_efficiency -
        #                 battery_discharge[i] / one_way_efficiency
        #                 ]
        constraints += [battery_energy[i + 1] == battery_energy[i] -
                        battery_power[i]]

    # Objective
    old_retail_bill = sum(cp.multiply(energy_price, load))
    new_retail_bill = sum(cp.multiply(energy_price, net_load_after_storage))

    battery_cost = sum(cp.multiply(energy_price, battery_power))
    # battery_charge_cost = sum(cp.multiply(energy_price, battery_charge))
    # battery_discharge_revenue = sum(cp.multiply(energy_price, battery_discharge))

    battery_export_revenue = sum(cp.multiply(energy_price, battery_discharge_to_grid))

    objective = cp.Maximize(battery_cost)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        print('Using GLPK_MI solver')
        result = problem.solve(solver=cp.GLPK_MI, verbose=False)
    except:
        print('Using ECOS_BB solver')
        result = problem.solve(solver=cp.ECOS_BB, verbose=False)

    # Check the problem status and determine the final demand target
    if problem.status not in ["infeasible", "unbounded"]:
        print('Final bill is {}'.format(problem.value))
        print(data_horizon["Datetime (he)"][0], battery_charge_cost.value, battery_discharge_revenue.value, battery_export_revenue.value,
              old_retail_bill.value, new_retail_bill.value)
        battery_soc = battery_energy.value / battery_size_kWh
        battery_charge = battery_charge.value
        battery_discharge_to_load = battery_discharge_to_load.value
        battery_discharge_to_grid = battery_discharge_to_grid.value
        export_yes = battery_discharge_to_grid >= 0
        net_load_after_storage = net_load_after_storage.value
    else:
        print('Problem {}'.format(problem.status))

    return battery_charge, battery_discharge_to_load, battery_discharge_to_grid, export_yes, battery_soc, net_load_after_storage


SVet_absolute_path = "/Applications/storagevet2v101/StorageVET-master-git/"
default_params_file = "Model_Parameters_2v1-0-2_default_03-2021.csv"
params = pd.read_csv(SVet_absolute_path + default_params_file)

battery_size = float(params.loc[(params.Key == "ene_max_rated") & (params.Tag == "Battery"), 'Value'])
battery_power = float(params.loc[(params.Key == "ch_max_rated") & (params.Tag == "Battery"), 'Value'])
one_way_efficiency = np.sqrt(float(params.loc[(params.Key == "rte") & (params.Tag == "Battery"), 'Value']) / 100)
start_soc = float(params.loc[(params.Key == "soc_target") & (params.Tag == "Battery"), 'Value']) / 100
max_soc = float(params.loc[(params.Key == "ulsoc") & (params.Tag == "Battery"), 'Value']) / 100
min_soc = float(params.loc[(params.Key == "llsoc") & (params.Tag == "Battery"), 'Value']) / 100

# n_horizon = int(params.loc[(params.Key == "n") & (params.Tag == "Scenario"), 'Value'])
# n_control = int(params.loc[(params.Key == "n_control") & (params.Tag == "Scenario"), 'Value'])
n_horizon = 48
n_control = 24

raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_pjm_2019.csv")
raw_data["Datetime (he)"] = pd.to_datetime(raw_data["Datetime (he)"])

data = raw_data.copy(deep=True)
start_datetime_he = data["Datetime (he)"][0]

raw_data_extra_days = raw_data.copy(deep=True).head(n_horizon)
raw_data_extra_days["Datetime (he)"] = raw_data_extra_days["Datetime (he)"] + pd.offsets.DateOffset(years=1)
raw_data = raw_data.append(raw_data_extra_days).reset_index(drop=True)

datetime_he_mpc = len(data["Datetime (he)"]) * [0]
load_mpc = len(data["Datetime (he)"]) * [0]
lmp_mpc = len(data["Datetime (he)"]) * [0]
soc_mpc = len(data["Datetime (he)"]) * [0]
charge_mpc = len(data["Datetime (he)"]) * [0]
discharge_to_load_mpc = len(data["Datetime (he)"]) * [0]
discharge_to_grid_mpc = len(data["Datetime (he)"]) * [0]
export_yes_mpc = len(data["Datetime (he)"]) * [0]
net_load_mpc = len(data["Datetime (he)"]) * [0]

for i in range(len(data["Datetime (he)"])):
    if data["Datetime (he)"][i] == start_datetime_he:
        end_datetime_he_horizon = start_datetime_he + timedelta(hours=n_horizon - 1)
        data_horizon = raw_data.loc[(raw_data["Datetime (he)"] >= start_datetime_he) &
                                    (raw_data["Datetime (he)"] <= end_datetime_he_horizon)].reset_index(drop=True)

        # Apply mpc_mip control optimization
        battery_charge, battery_discharge_to_load, battery_discharge_to_grid, export_yes, soc_hs, net_load_after_storage = \
            run_optimization(daily_load=data_horizon.iloc[0:n_horizon],
                             energy_price=data_horizon["DA Price ($/kWh)"][0:n_horizon],
                             battery_size_kWh=battery_size,
                             battery_power_kW=battery_power,
                             min_soc=min_soc, max_soc=max_soc,
                             current_soc=start_soc,
                             one_way_efficiency=one_way_efficiency)

        # Append results
        datetime_he_mpc[i:i + n_control] = data_horizon["Datetime (he)"][0:n_control]
        load_mpc[i:i + n_control] = data_horizon["Site Load (kW)"][0:n_control]
        lmp_mpc[i:i + n_control] = data_horizon["DA Price ($/kWh)"][0:n_control]
        soc_mpc[i:i + n_control] = soc_hs[0:n_control]
        charge_mpc[i:i + n_control] = battery_charge[0:n_control]
        discharge_to_load_mpc[i:i + n_control] = battery_discharge_to_load[0:n_control]
        discharge_to_grid_mpc[i:i + n_control] = battery_discharge_to_grid[0:n_control]
        export_yes_mpc[i:i + n_control] = export_yes[0:n_control]
        net_load_mpc[i:i + n_control] = net_load_after_storage[0:n_control]

        # Update start time and soc for mpc_mip
        start_datetime_he = start_datetime_he + timedelta(hours=n_control)
        start_soc = soc_hs[n_control]

    else:
        pass

results = pd.DataFrame()
results["Datetime (he)"] = datetime_he_mpc
results["Site Load (kW)"] = load_mpc
results["DA Price ($/kWh)"] = lmp_mpc
results["soc_hs"] = soc_mpc
results["charge_mpc"] = charge_mpc
results["discharge_to_load_mpc"] = discharge_to_load_mpc
results["discharge_to_grid_mpc"] = discharge_to_grid_mpc
results["export_yes_mpc"] = export_yes_mpc
results["net_load"] = net_load_mpc
results.to_csv("results_lp.csv")

# Output results
results_dict = {}
results_dict["old_retail_bill"] = - np.sum(results["DA Price ($/kWh)"] * results["Site Load (kW)"])
results_dict["new_retail_bill"] = - np.sum(results["DA Price ($/kWh)"] * results["net_load"])
results_dict["retail_bill_savings"] = results_dict["new_retail_bill"] - results_dict["old_retail_bill"]
results_dict["battery_charge_cost"] = np.sum(results["DA Price ($/kWh)"] * results["charge_mpc"])
results_dict["battery_retail_revenues"] = np.sum(results["DA Price ($/kWh)"] * results["discharge_to_load_mpc"])
results_dict["battery_export_revenue"] = np.sum(results["DA Price ($/kWh)"] * results["discharge_to_grid_mpc"])
results_dict["battery_retail_profits"] = results_dict["battery_retail_revenues"] + results_dict["battery_charge_cost"]
results_dict["battery_total_profits"] = results_dict["battery_export_revenue"] + \
                                        results_dict["battery_retail_revenues"] + results_dict["battery_charge_cost"]
with open('results_lp.json', 'w') as f:
    json.dump(results_dict, f)
