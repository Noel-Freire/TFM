from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import plotly.express as px
import numpy as np
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd

def plot_acf_pacf(df, lags):
    acf_values=[]
    pacf_values=[]
    acf_values = acf(df, nlags=24)
    pacf_values = pacf(df, nlags=24, method='ywm')
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    sm.graphics.tsa.plot_acf(df, lags=lags, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df, lags=lags, ax=ax[1], method='ywm')  # método robusto
    return fig,acf_values,pacf_values




def fft(df,frecuencia):
    Muestras=len(df)
    Muestras_hora=1
    #transformada de fourier
    tf=abs(np.fft.rfft(df,Muestras))
    #Vector Frecuencias, eje hor
    frec=np.fft.rfftfreq(Muestras,d=1./Muestras_hora)
    #Grafica
    if frecuencia=='M':
        tiempo='mes'
        mult=12
        tiempo2='año'
    elif frecuencia=='D':
        tiempo='dia'
        mult=30
        tiempo2='mes'
    elif frecuencia=='H':
        tiempo='hora'
        mult=24
        tiempo2='dia'
    elif frecuencia=='Y':
        tiempo='año'
        mult=4
        tiempo2='4 años'
    else:
        tiempo='minuto'
        mult=60
        tiempo2='hora'
    fig=px.line(x=frec,y=tf,log_x=True)
    fig.update_layout(xaxis_title=f'Frecuencia (1/{tiempo})')
    # Excluimos la frecuencia 0 (DC)
    tf_no_dc = tf[1:]
    frec_no_dc = frec[1:]
    # Indices de los 3 valores más altos (ordenados de mayor a menor)
    top_3_indices = np.argsort(tf_no_dc)[-3:][::-1]  # Últimos 3, luego invertir
    # Obtener las frecuencias y amplitudes correspondientes
    top_3_frecs = frec_no_dc[top_3_indices]

    return fig,top_3_frecs,tiempo,mult,tiempo2




def test_adfuller(Datos,lags,regression):
    X = Datos.values
    result = adfuller(X,maxlag=lags,regression=regression)
    return result




def test_kpss (Datos,lags,regression):
    kpss_result=kpss(Datos.values, regression=regression, nlags=lags)
    return kpss_result




def descompose_seasonal(Datos, objetivo, lags):

    if isinstance(Datos.index, pd.PeriodIndex):
        Datos.index = Datos.index.to_timestamp()
    decomposition_result = seasonal_decompose(Datos[objetivo], model="additive", period=lags) 

    fig = make_subplots(rows=4, cols=1, subplot_titles=("Original Series", "Trend Component", "Seasonal Component", "Residuals"))

    fig.add_trace(go.Scatter(x=Datos.index, y=decomposition_result.observed, mode="lines", name="Original", line=dict(color="darkblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=decomposition_result.trend, mode="lines", name="Trend", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=decomposition_result.seasonal, mode="lines", name="Seasonality", line=dict(color="purple")), row=3, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=decomposition_result.resid, mode="lines", name="Residuals", line=dict(color="orange")), row=4, col=1)

    fig.update_layout(title="Decomposition of Time Series", height=900, showlegend=False, plot_bgcolor="whitesmoke")

    animation_frames = []
    step = max(1, int(len(Datos) / 50))
    for i in range(0,len(Datos),int(step)):
        frame = go.Frame(data=[go.Scatter(x=Datos.index[:i+1], y=decomposition_result.observed[:i+1], mode="lines", name="Original", line=dict(color="darkblue")),
                                go.Scatter(x=Datos.index[:i+1], y=decomposition_result.trend[:i+1], mode="lines", name="Trend", line=dict(color="green")),
                                go.Scatter(x=Datos.index[:i+1], y=decomposition_result.seasonal[:i+1], mode="lines", name="Seasonality", line=dict(color="purple")),
                                go.Scatter(x=Datos.index[:i+1], y=decomposition_result.resid[:i+1], mode="lines", name="Residuals", line=dict(color="orange"))],
                        traces=[0, 1, 2, 3])
        animation_frames.append(frame)

    fig.frames = animation_frames

    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False,
                                        buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                                                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])])
    return fig




def seasonal_SLT(Datos, fecha, objetivo, lags):
    if isinstance(Datos.index, pd.PeriodIndex):
        Datos.index = Datos.index.to_timestamp()
        
    stl1 = STL(Datos[objetivo], period=lags)  
    result = stl1.fit()
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Original Series", "Trend Component", "Seasonal Component", "Residuals"))

    fig.add_trace(go.Scatter(x=Datos.index, y=Datos[objetivo], mode="lines", name="Original", line=dict(color="darkblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=result.trend, mode="lines", name="Trend", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=result.seasonal, mode="lines", name="Seasonality", line=dict(color="purple")), row=3, col=1)
    fig.add_trace(go.Scatter(x=Datos.index, y=result.resid, mode="lines", name="Residuals", line=dict(color="orange")), row=4, col=1)

    fig.update_layout(title="Decomposition of Time Series", height=900, showlegend=False, plot_bgcolor="whitesmoke")

    animation_frames = []
    step = max(1, int(len(Datos) / 50))
    for i in range(0,len(Datos),int(step)):
        frame = go.Frame(data=[go.Scatter(x=Datos.index[:i+1], y=result.observed[:i+1], mode="lines", name="Original", line=dict(color="darkblue")),
                                go.Scatter(x=Datos.index[:i+1], y=result.trend[:i+1], mode="lines", name="Trend", line=dict(color="green")),
                                go.Scatter(x=Datos.index[:i+1], y=result.seasonal[:i+1], mode="lines", name="Seasonality", line=dict(color="purple")),
                                go.Scatter(x=Datos.index[:i+1], y=result.resid[:i+1], mode="lines", name="Residuals", line=dict(color="orange"))],
                        traces=[0, 1, 2, 3])
        animation_frames.append(frame)

    fig.frames = animation_frames

    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False,
                                        buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                                                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])])
    return fig

