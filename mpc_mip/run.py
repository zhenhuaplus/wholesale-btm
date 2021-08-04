import pandas as pd
import numpy as np
from datetime import timedelta
import cvxpy as cp
import json
import os

from mpc_mip.financial_analysis import calculate_finance


def run_optimization(daily_load, load_price, charge_price, discharge_price, export_price,
                     battery_size_kWh, battery_power_kW, min_soc, max_soc, current_soc, one_way_efficiency,
                     demand_charge, peak_load_so_far, peak_net_load_so_far):
    # Obtain parameters
    battery_energy_min = battery_size_kWh * min_soc
    battery_energy_max = battery_size_kWh * max_soc

    # Design optimization-based control policies
    # Define variables
    load = daily_load["Site Load (kW)"]
    net_load_after_storage = cp.Variable(daily_load.shape[0])
    battery_charge = cp.Variable(daily_load.shape[0])
    battery_discharge_to_load = cp.Variable(daily_load.shape[0])
    battery_discharge_to_grid = cp.Variable(daily_load.shape[0])
    is_exporting = cp.Variable(daily_load.shape[0], boolean=True)
    is_charging = cp.Variable(daily_load.shape[0], boolean=True)
    battery_energy = cp.Variable(daily_load.shape[0] + 1)

    # Initialize constraints
    constraints = []

    # General net load and battery power constraints
    constraints += [net_load_after_storage == np.array(load) - battery_charge - battery_discharge_to_load]
    constraints += [0 <= battery_discharge_to_load,
                    battery_discharge_to_load <= cp.multiply(cp.minimum(battery_power_kW, load), 1 - is_charging)]
    constraints += [0 <= battery_discharge_to_grid,
                    battery_discharge_to_grid <= cp.multiply(battery_power_kW, 1 - is_charging)]
    constraints += [battery_discharge_to_load + battery_discharge_to_grid <= battery_power_kW]
    constraints += [cp.multiply(- battery_power_kW, is_charging) <= battery_charge, battery_charge <= 0]

    # # Battery charge or discharge cannot be greater than 0 at the same time (does not follow DCP rules if on)
    # constraints += [sum(cp.multiply(battery_discharge_to_load, battery_charge)) == 0]
    # constraints += [sum(cp.multiply(battery_discharge_to_grid, battery_charge)) == 0]

    # Export constraints
    export_to_grid_allowed = True
    if export_to_grid_allowed:
        constraints += [battery_discharge_to_load >= cp.multiply(load, is_exporting)]
        constraints += [battery_discharge_to_grid <= cp.multiply((battery_power_kW - load).clip(lower=0), is_exporting)]
    else:
        constraints += [is_exporting == 0]
        constraints += [battery_discharge_to_grid == 0]

    # Battery energy dynamics
    constraints += [battery_energy_min <= battery_energy, battery_energy <= battery_energy_max]
    constraints += [battery_energy[0] == battery_size_kWh * current_soc]
    for i in range(daily_load.shape[0]):
        constraints += [battery_energy[i + 1] == battery_energy[i] -
                        battery_charge[i] * one_way_efficiency -
                        battery_discharge_to_load[i] / one_way_efficiency -
                        battery_discharge_to_grid[i] / one_way_efficiency]

    # Objective
    # Calculate retail bills
    optimize_for_demand_saving = True
    if optimize_for_demand_saving:
        peak_load = cp.maximum(cp.max(load), peak_load_so_far)
        old_demand_charge = - peak_load * demand_charge
        peak_net_load = cp.maximum(cp.max(net_load_after_storage), peak_net_load_so_far)
        new_demand_charge = - peak_net_load * demand_charge
        demand_charge_savings = new_demand_charge - old_demand_charge
    else:
        demand_charge_savings = 0
    old_energy_charge = - sum(cp.multiply(load_price, load))
    new_energy_charge = - sum(cp.multiply(load_price, net_load_after_storage))
    energy_charge_savings = new_energy_charge - old_energy_charge
    # Calculate battery bills
    battery_charge_cost = sum(cp.multiply(charge_price, battery_charge))
    battery_discharge_revenue = sum(cp.multiply(discharge_price, battery_discharge_to_load))
    battery_export_revenue = sum(cp.multiply(export_price, battery_discharge_to_grid))
    battery_profits = battery_charge_cost + battery_discharge_revenue + battery_export_revenue
    # Set total objective
    objective = cp.Maximize(demand_charge_savings + battery_profits)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        print('Using GLPK_MI solver')
        problem.solve(solver=cp.GLPK_MI, verbose=False)
    except:
        print('Using ECOS_BB solver')
        problem.solve(solver=cp.ECOS_BB, verbose=False)

    # Check the problem status and determine the final demand target
    if problem.status not in ["infeasible", "unbounded"]:
        print("{} charge:{} discharge:{} export:{} old energy bill:{} new energy bill:{}".format(
            daily_load["Datetime (he)"][0],
            round(battery_charge_cost.value, 2),
            round(battery_discharge_revenue.value, 2),
            round(battery_export_revenue.value, 2),
            round(old_energy_charge.value, 2),
            round(new_energy_charge.value, 2)))
        dispatch_results = {
            "battery_soc": battery_energy.value / battery_size_kWh,
            "battery_charge": battery_charge.value,
            "battery_discharge_to_load": battery_discharge_to_load.value,
            "battery_discharge_to_grid": battery_discharge_to_grid.value,
            "is_exporting": is_exporting.value,
            "is_charging": is_charging.value,
            "net_load_after_storage": net_load_after_storage.value
        }
    else:
        print('Problem {}'.format(problem.status))
        dispatch_results = None

    return dispatch_results


def store_results(results, results_dict):

    # Check if run log csv has been created, if not create a new csv with relevant column names
    run_log_file = "results/log.csv"
    log_columns = ["runID"] + list(results_dict["key_params"].keys()) + list(results_dict["savings"].keys())
    if not (os.path.exists(run_log_file)):
        runs_log = pd.DataFrame(columns=log_columns, index=None)
        runs_log.to_csv(run_log_file, index=None)
    else:
        runs_log = pd.read_csv(run_log_file)

    # Create entry in runs log file
    runID = str(int(runs_log['runID'].max()) + 1) if runs_log['runID'].max() is not np.nan else str(1)
    log_values = [runID] + list(results_dict["key_params"].values()) + list(results_dict["savings"].values())
    runs_log = runs_log.append(pd.Series(log_values, index=runs_log.columns),
                               ignore_index=True)
    runs_log.to_csv(run_log_file, index=None)

    # Save results in csv and json files
    results.to_csv("results/result_dispatches_{}.csv".format(runID))
    with open('results/result_bills_{}.json'.format(runID), 'w') as f:
        json.dump(results_dict, f)


def run_mpc(raw_data, location, customer_type, export_price_type,
            battery_size_kWh, battery_power_kW, min_soc, max_soc, one_way_efficiency, demand_charge,
            start_soc, n_horizon, n_control):
    raw_data["Datetime (he)"] = pd.to_datetime(raw_data["Datetime (he)"])
    data = raw_data.copy(deep=True)
    start_datetime_he = data["Datetime (he)"][0]
    initial_start_soc = start_soc

    raw_data_extra_days = raw_data.copy(deep=True).head(n_horizon)
    raw_data_extra_days["Datetime (he)"] = raw_data_extra_days["Datetime (he)"] + pd.offsets.DateOffset(years=1)
    raw_data = raw_data.append(raw_data_extra_days).reset_index(drop=True)

    datetime_he_mpc, load_mpc, load_price_mpc, charge_price_mpc, discharge_price_mpc, export_price_mpc, \
    soc_mpc, charge_mpc, discharge_to_load_mpc, discharge_to_grid_mpc, is_exporting_mpc, \
    is_charging_mpc, net_load_mpc = \
        (len(data["Datetime (he)"]) * [0] for i in range(13))

    for i in range(len(data["Datetime (he)"])):
        if data["Datetime (he)"][i] == start_datetime_he:

            # Obtain perfect knowledge about the next horizon
            end_datetime_he_horizon = start_datetime_he + timedelta(hours=n_horizon - 1)
            data_horizon = raw_data.loc[(raw_data["Datetime (he)"] >= start_datetime_he) &
                                        (raw_data["Datetime (he)"] <= end_datetime_he_horizon)].reset_index(drop=True)

            # Obtain peak load and peak net load so far for this month
            datetime_he_mpc_valid = [datetime for datetime in datetime_he_mpc if datetime != 0]
            try:
                idx = [idx for idx in range(len(datetime_he_mpc_valid))
                       if (datetime_he_mpc_valid[idx] - timedelta(hours=1)).month == start_datetime_he.month]
                peak_load_so_far = max(load_mpc[idx[0]:idx[-1]])
                peak_net_load_so_far = max(net_load_mpc[idx[0]:idx[-1]])
            except:
                peak_load_so_far = 0
                peak_net_load_so_far = 0

            # Add noise to perfect knowledge or add error correction to predicted values if needed
            # TODO
            load_price_pred = data_horizon["load_price"][0:n_horizon]
            charge_price_pred = data_horizon["charge_price"][0:n_horizon]
            discharge_price_pred = data_horizon["discharge_price"][0:n_horizon]
            export_price_pred = data_horizon["export_price"][0:n_horizon]

            # Apply mpc_mip control optimization
            dispatch_results = run_optimization(daily_load=data_horizon.iloc[0:n_horizon],
                                                load_price=load_price_pred, charge_price=charge_price_pred,
                                                discharge_price=discharge_price_pred, export_price=export_price_pred,
                                                battery_size_kWh=battery_size_kWh, battery_power_kW=battery_power_kW,
                                                min_soc=min_soc, max_soc=max_soc, current_soc=start_soc,
                                                one_way_efficiency=one_way_efficiency,
                                                demand_charge=demand_charge,
                                                peak_load_so_far=peak_load_so_far,
                                                peak_net_load_so_far=peak_net_load_so_far)

            # Append results
            datetime_he_mpc[i:i + n_control] = data_horizon["Datetime (he)"][0:n_control]
            load_mpc[i:i + n_control] = data_horizon["Site Load (kW)"][0:n_control]
            load_price_mpc[i:i + n_control] = load_price_pred[0:n_control]
            charge_price_mpc[i:i + n_control] = charge_price_pred[0:n_control]
            discharge_price_mpc[i:i + n_control] = discharge_price_pred[0:n_control]
            export_price_mpc[i:i + n_control] = export_price_pred[0:n_control]

            soc_mpc[i:i + n_control] = dispatch_results["battery_soc"][0:n_control]
            charge_mpc[i:i + n_control] = dispatch_results["battery_charge"][0:n_control]
            discharge_to_load_mpc[i:i + n_control] = dispatch_results["battery_discharge_to_load"][0:n_control]
            discharge_to_grid_mpc[i:i + n_control] = dispatch_results["battery_discharge_to_grid"][0:n_control]
            is_exporting_mpc[i:i + n_control] = dispatch_results["is_exporting"][0:n_control]
            is_charging_mpc[i:i + n_control] = dispatch_results["is_charging"][0:n_control]
            net_load_mpc[i:i + n_control] = dispatch_results["net_load_after_storage"][0:n_control]

            # Update start time and soc for mpc_mip
            start_datetime_he = start_datetime_he + timedelta(hours=n_control)
            start_soc = dispatch_results["battery_soc"][n_control]

        else:
            pass

    results = pd.DataFrame()
    results["Datetime (hs)"] = np.array(datetime_he_mpc) - timedelta(hours=1)
    results["month (hs)"] = pd.to_datetime(results["Datetime (hs)"]).dt.month
    results["Datetime (he)"] = datetime_he_mpc

    results["load_price"] = load_price_mpc
    results["charge_price"] = charge_price_mpc
    results["discharge_price"] = discharge_price_mpc
    results["export_price"] = export_price_mpc
    results["soc_hs"] = soc_mpc
    results["load"] = load_mpc
    results["charge_mpc"] = charge_mpc
    results["discharge_to_load"] = discharge_to_load_mpc
    results["discharge_to_grid"] = discharge_to_grid_mpc
    results["net_load"] = net_load_mpc
    results["net_load_after_export"] = np.array(net_load_mpc) - np.array(discharge_to_grid_mpc)
    results["is_exporting"] = is_exporting_mpc
    results["is_charging"] = is_charging_mpc

    # Output results
    results_dict = {
        "key_params": {
            "location": location,
            "customer_type": customer_type,
            "demand_charge": demand_charge,
            "export_price": raw_data["export_price"][0],
            "export_price_type": export_price_type,
            "battery_size_kWh": battery_size_kWh,
            "battery_power_kW": battery_power_kW,
            "min_soc": min_soc,
            "max_soc": max_soc,
            "one_way_efficiency": one_way_efficiency,
            "start_soc": initial_start_soc,
            "n_horizon": n_horizon,
            "n_control": n_control
        },
        "savings": {}
    }

    results_dict["savings"]["old_energy_charge"] = - np.sum(results["load_price"] * results["load"])
    results_dict["savings"]["new_energy_charge"] = - np.sum(results["load_price"] * results["net_load"])
    results_dict["savings"]["energy_charge_savings"] = results_dict["savings"]["new_energy_charge"] - \
                                                       results_dict["savings"]["old_energy_charge"]

    results_dict["savings"]["old_demand_charge"] = - np.sum(results.groupby("month (hs)")["load"].agg("max")) * \
                                                   demand_charge
    results_dict["savings"]["new_demand_charge"] = - np.sum(results.groupby("month (hs)")["net_load"].agg("max")) * \
                                                   demand_charge
    results_dict["savings"]["demand_charge_savings"] = results_dict["savings"]["new_demand_charge"] - \
                                                       results_dict["savings"]["old_demand_charge"]

    results_dict["savings"]["retail_profits"] = results_dict["savings"]["energy_charge_savings"] + \
                                                results_dict["savings"]["demand_charge_savings"]

    results_dict["savings"]["battery_charge_cost"] = np.sum(results["charge_price"] * results["charge_mpc"])
    results_dict["savings"]["battery_discharge_revenues"] = np.sum(results["discharge_price"] *
                                                                   results["discharge_to_load"])
    results_dict["savings"]["battery_retail_profits"] = results_dict["savings"]["battery_discharge_revenues"] + \
                                                        results_dict["savings"]["battery_charge_cost"]
    results_dict["savings"]["battery_export_revenue"] = np.sum(results["export_price"] *
                                                               results["discharge_to_grid"])
    results_dict["savings"]["total_profits"] = results_dict["savings"]["battery_retail_profits"] +\
                                               results_dict["savings"]["battery_export_revenue"] + \
                                               results_dict["savings"]["demand_charge_savings"]
    results_dict["savings"]["irr"] = calculate_finance(battery_size_kWh=battery_size_kWh,
                                                       battery_total_profits_per_year=results_dict["savings"]["total_profits"])[0]
    results_dict["savings"]["npv"] = calculate_finance(battery_size_kWh=battery_size_kWh,
                                                       battery_total_profits_per_year=results_dict["savings"]["total_profits"])[1]

    # Store results
    store_results(results, results_dict)

    return results, results_dict









