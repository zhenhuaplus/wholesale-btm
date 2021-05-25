import pandas as pd
import numpy as np
from datetime import timedelta
import cvxpy as cp


def run_optimization(daily_load, energy_price,
                     battery_size_kWh, battery_power_kW,
                     min_soc, max_soc, current_soc, one_way_efficiency):
    # Obtain parameters
    battery_energy_min = battery_size_kWh * min_soc
    battery_energy_max = battery_size_kWh * max_soc

    # Design optimization-based control policies
    # Define variables
    load = cp.Variable(daily_load.shape[0])
    net_load_after_storage = cp.Variable(daily_load.shape[0])
    battery_charge = cp.Variable(daily_load.shape[0])
    battery_discharge = cp.Variable(daily_load.shape[0])
    battery_energy = cp.Variable(daily_load.shape[0] + 1)

    # Initialize constraints
    constraints = []

    # Battery power/energy constraint
    constraints += [0 <= battery_charge, battery_charge <= battery_power_kW]
    constraints += [-battery_power_kW <= battery_discharge, battery_discharge <= 0]
    constraints += [battery_energy_min <= battery_energy, battery_energy <= battery_energy_max]

    # Starting/ending SOC constraint
    constraints += [battery_energy[0] == battery_size_kWh * current_soc]
    # constraints += [battery_energy[-1] == battery_size_kWh * current_soc]

    # Net load constraint
    constraints += [net_load_after_storage >= 0]
    constraints += [net_load_after_storage == load + battery_charge + battery_discharge]
    for i in range(daily_load.shape[0]):
        constraints += [load[i] == daily_load["Site Load (kW)"].iloc[i]]

    # Battery energy dynamics
    for i in range(daily_load.shape[0]):
        constraints += [battery_energy[i + 1] == battery_energy[i] +
                        battery_charge[i] * one_way_efficiency +
                        battery_discharge[i] / one_way_efficiency]

    # Objective
    objective = cp.Minimize(sum(cp.multiply(energy_price, net_load_after_storage)))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        print('Using GLPK_MI solver')
        result = problem.solve(solver=cp.GLPK_MI, verbose=False)
    except:
        try:
            print('Using ECOS solver')
            result = problem.solve(solver=cp.ECOS, verbose=False)
        except:
            print('Using SCS solver')
            result = problem.solve(solver=cp.SCS, verbose=False)

    # Check the problem status and determine the final demand target
    if problem.status not in ["infeasible", "unbounded"]:
        print('Final bill is {}'.format(problem.value))
        battery_power = - (battery_charge.value + battery_discharge.value)
        battery_soc = battery_energy.value / battery_size_kWh
    else:
        print('Problem {}'.format(problem.status))

    return battery_power, battery_soc


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
n_horizon = 24
n_control = 12

raw_data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_pjm_2019_200x.csv")
raw_data["Datetime (he)"] = pd.to_datetime(raw_data["Datetime (he)"])
data = pd.read_csv("/Users/zhenhua/Desktop/price_data/hourly_timeseries_pjm_2019_200x.csv").head(48)
data["Datetime (he)"] = pd.to_datetime(data["Datetime (he)"])
start_datetime_he = data["Datetime (he)"][0]

datetime_he_mpc = len(data["Datetime (he)"]) * [0]
load_mpc = len(data["Datetime (he)"]) * [0]
lmp_mpc = len(data["Datetime (he)"]) * [0]
soc_mpc = len(data["Datetime (he)"]) * [0]
power_mpc = len(data["Datetime (he)"]) * [0]

for i in range(len(data["Datetime (he)"])):
    if data["Datetime (he)"][i] == start_datetime_he:
        end_datetime_he_horizon = start_datetime_he + timedelta(hours=n_horizon - 1)
        data_horizon = data.loc[(raw_data["Datetime (he)"] >= start_datetime_he) &
                                (raw_data["Datetime (he)"] <= end_datetime_he_horizon)].reset_index(drop=True)

        # Apply mpc control optimization
        power, soc_hs = run_optimization(daily_load=data_horizon.iloc[0:n_horizon],
                                         energy_price=data_horizon["DA Price ($/kWh)"][0:n_horizon],
                                         battery_size_kWh=battery_size, battery_power_kW=battery_power,
                                         min_soc=min_soc, max_soc=max_soc, current_soc=start_soc,
                                         one_way_efficiency=one_way_efficiency)

        # Append results
        datetime_he_mpc[i:i + n_control] = data_horizon["Datetime (he)"][0:n_control]
        load_mpc[i:i + n_control] = data_horizon["Site Load (kW)"][0:n_control]
        lmp_mpc[i:i + n_control] = data_horizon["DA Price ($/kWh)"][0:n_control]
        soc_mpc[i:i + n_control] = soc_hs[0:n_control]
        power_mpc[i:i + n_control] = power[0:n_control]

        # Update start time and soc for mpc
        start_datetime_he = start_datetime_he + timedelta(hours=n_control)
        start_soc = soc_hs[n_control]

    else:
        pass

results = pd.DataFrame()
results["Datetime (he)"] = datetime_he_mpc
results["Site Load (kW)"] = load_mpc
results["DA Price ($/kWh)"] = lmp_mpc
results["soc_hs"] = soc_mpc
results["power"] = power_mpc
results.to_csv("test.csv")
