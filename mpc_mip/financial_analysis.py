import numpy as np


def calculate_finance(battery_size_kWh, battery_total_profits_per_year):
    # Read multiyear outputs excel file
    project_years = 15
    depreciation_years = 15
    salvage_value_ratio = 0
    battery_degradation_per_year = 0.03
    discount_rate = 0.06

    ess_cost_dollar_per_kwh = 300  # 300~400
    operation_cost_dollar_per_kwh = 2  # 2~3.7
    epc_cost_dollar_per_kwh = 0
    combined_tax_rate = 0.21

    # Dispatch and saving results
    available_capacity_kwh = np.ones(project_years) * battery_size_kWh
    project_revenues = np.zeros(project_years)
    for i in range(project_years):
        project_revenues[i] = battery_total_profits_per_year * (1 - battery_degradation_per_year * i)
        available_capacity_kwh[i] = available_capacity_kwh[i] * (1 - battery_degradation_per_year * i)

    # Investment and operations calculations
    ess_cost = - battery_size_kWh * ess_cost_dollar_per_kwh
    epc_cost = - battery_size_kWh * epc_cost_dollar_per_kwh
    project_investment_cost = ess_cost + epc_cost
    operation_cost = - available_capacity_kwh * operation_cost_dollar_per_kwh
    ebitda = np.array(project_revenues) + np.array(operation_cost)

    # Tax and depreciation calculations
    depreciation = [0] * project_years
    for i in range(depreciation_years):
        depreciation[i] = project_investment_cost * (1 - salvage_value_ratio) / depreciation_years
    full_taxable_profit = np.array(ebitda) + np.array(depreciation)
    full_income_tax = [- max(0, i * combined_tax_rate) for i in full_taxable_profit]
    full_net_profit = np.array(full_taxable_profit) + np.array(full_income_tax)

    # Cash flow and IRR calculations
    full_cash_flow = - np.array(depreciation) + np.array(full_net_profit)
    full_cash_flow = [project_investment_cost] + list(full_cash_flow)
    full_irr = np.round(np.irr(full_cash_flow), 5)
    full_npv = np.npv(discount_rate, full_cash_flow).round(3)

    return [full_irr, full_npv]
