import numpy as np
from random import gauss
import plotly.graph_objects as go
from arch import arch_model

# Step 1: Generate synthetic GARCH(2,2) data
n = 1000
w = 0.5
a1, a2 = 0.1, 0.2
b1, b2 = 0.3, 0.4
split = int(n * 0.1)

X = [gauss(0, 1), gauss(0, 1)]
V = [1, 1]

for _ in range(n):
    v_new = np.sqrt(w + a1 * X[-1]**2 + a2 * X[-2]**2 + b1 * V[-1]**2 + b2 * V[-2]**2)
    x_new = gauss(0, 1) * v_new
    V.append(v_new)
    X.append(x_new)

# Step 2: Split data into training and testing sets
df_train = X[:-split]
df_test = X[-split:]

# Step 3: Fit GARCH(2,2) model on training data
mdl = arch_model(df_train, p=2, q=2)
fit = mdl.fit(disp='off')

# Step 4: Generate forecasts
pred_short = fit.forecast(horizon=split)
pred_long = fit.forecast(horizon=1000)
V_short = np.sqrt(pred_short.variance.values[-1, :])
V_long = np.sqrt(pred_long.variance.values[-1, :])

# Step 5: Perform rolling forecast
roll_V = []
for i in range(split):
    tmp_train = X[:-(split - i)]
    tmp_mdl = arch_model(tmp_train, p=2, q=2)
    tmp_fit = tmp_mdl.fit(disp='off')
    tmp_pred = tmp_fit.forecast(horizon=1)
    roll_V.append(np.sqrt(tmp_pred.variance.values[-1, 0]))

# Step 6: Plot data and volatility
fig1 = go.Figure()
fig1.add_trace(go.Scatter(y=X, mode='lines', line=dict(color='#FF6B6B'), name='Data'))
fig1.add_trace(go.Scatter(y=V, mode='lines', line=dict(color='#4ECDC4'), name='Volatility'))
fig1.update_layout(
    title='Data and Volatility', title_font_color='white',
    plot_bgcolor='rgb(40, 40, 40)', paper_bgcolor='rgb(40, 40, 40)',
    font_color='white', showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5)
)

# Step 7: Plot short-term volatility prediction
fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=V[-split:], mode='lines', line=dict(color='#FF6B6B'), name='True Volatility'))
fig2.add_trace(go.Scatter(y=V_short, mode='lines', line=dict(color='#4ECDC4', dash='dash'), name='Predicted Volatility'))
fig2.update_layout(
    title='Short-Term Volatility Prediction', title_font_color='white',
    plot_bgcolor='rgb(40, 40, 40)', paper_bgcolor='rgb(40, 40, 40)',
    font_color='white', showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5)
)

# Step 8: Plot long-term volatility prediction
fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=V[-split:], mode='lines', line=dict(color='#FF6B6B'), name='True Volatility'))
fig3.add_trace(go.Scatter(y=V_long, mode='lines', line=dict(color='#4ECDC4', dash='dash'), name='Predicted Volatility'))
fig3.update_layout(
    title='Long-Term Volatility Prediction', title_font_color='white',
    plot_bgcolor='rgb(40, 40, 40)', paper_bgcolor='rgb(40, 40, 40)',
    font_color='white', showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5)
)

# Step 9: Plot rolling forecast
fig4 = go.Figure()
fig4.add_trace(go.Scatter(y=V[-split:], mode='lines', line=dict(color='#FF6B6B'), name='True Volatility'))
fig4.add_trace(go.Scatter(y=roll_V, mode='lines', line=dict(color='#4ECDC4', dash='dash'), name='Rolling Prediction'))
fig4.update_layout(
    title='Rolling Volatility Prediction', title_font_color='white',
    plot_bgcolor='rgb(40, 40, 40)', paper_bgcolor='rgb(40, 40, 40)',
    font_color='white', showlegend=True,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5)
)

# Display plots
fig1.show()
fig2.show()
fig3.show()
fig4.show()