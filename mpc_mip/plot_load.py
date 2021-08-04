import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

load = pd.read_excel("results/daily_profile.xlsx")
color_discrete_sequence = ["#F58518", "#DC3912", "#FECB52", "#3366CC", "#B82E2E", "#316395", "#990099"]
customer_type_list = sorted(list(set(load["customer_type"])))

# load = load[(load["customer_type"] == "Office") | (load["customer_type"] == "School") |
#             (load["customer_type"] == "Hospital")]
# load = load[(load["customer_type"] == "Market") | (load["customer_type"] == "Restaurant")]
load = load[(load["customer_type"] == "Hotel") | (load["customer_type"] == "Apartment")]


fig = go.Figure()
for i in range(len(load)):
    customer_type = str(load.iloc[i, 2])
    customer_color = color_discrete_sequence[customer_type_list.index(customer_type)]
    data = np.array(load.iloc[i, 3:])

    fig.add_trace(go.Scatter(x=np.arange(0, 24, 1), y=data,
                             line=dict(color=customer_color, width=2))
                  )

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
    ),
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
    ),
    showlegend=False,
    plot_bgcolor='white',
    title="Group 3 daily average load profiles",
    xaxis_title="Hours",
    yaxis_title="Load (kW)",
    font=dict(family="Arial")
)
fig.update_yaxes(range=[0, 700])

fig.show()
