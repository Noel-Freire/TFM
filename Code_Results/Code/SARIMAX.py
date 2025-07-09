from Preprocesado import input_length
from Operaciones_datos import frecuencia_usuario
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

def crear_SARIMAX(df, tr_size, lags, Col_Predecir,seasonality,steps):
    train, val, test = input_length(df, tr_size, 0)
    prueba = pd.concat([val, test], ignore_index=False)
    modelo, mensaje = fit_sarimax_with_auto_arima(
    y=train[Col_Predecir].shift(-1).dropna(),  # Serie objetivo sin NaN
    X=train[:-1],    # Variables exógenas alineadas
    seasonal=seasonality, 
    m=lags, 
    verbose=True
    )
    # Predicción
    os.makedirs("SARIMAX", exist_ok=True)
    predicciones = []
    for i in range(len(prueba)):
        forecast = modelo.get_forecast(steps=1, exog=prueba[i:i+1])
        predicciones.append(forecast.predicted_mean.values[0])

    forecast_index = prueba.index[:steps]  # Convertir PeriodIndex a DatetimeIndex

    plt.figure(figsize=(10,5))

    # Graficar valores reales
    plt.plot(forecast_index, prueba[Col_Predecir][1:steps+1], label='Valores reales', marker='o')

    # Graficar predicciones
    plt.plot(forecast_index, predicciones[:steps], label='Predicciones', marker='o')

    # Graficar intervalo de confianza


    plt.legend()
    plt.title('Predicciones vs Valores Reales')
    plt.xlabel('Fecha')
    plt.ylabel('P')
    plt.savefig("SARIMAX/Ejemplo_SARIMAX.png", dpi=300)

    mae = mean_absolute_error(prueba['P'], predicciones)
    rmse = np.sqrt(mean_squared_error(prueba['P'], predicciones))
    mse=mean_squared_error(prueba['P'], predicciones)

 
    fig, axs = plt.subplots(1,1,figsize=(12,10))
    x = prueba['P']
    y = predicciones

    # Ajuste lineal: coeficientes [pendiente, intercepto]
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)

    # Dibujar puntos reales vs predichos
    axs.scatter(x, y, label=f'Datos t+1', alpha=0.5)

    # Dibujar recta de regresión
    x_line = np.linspace(np.min(x), np.max(x), 100)
    axs.plot(x_line, poly1d_fn(x_line), label=f'Recta t+1',color='red')

    # R2 para info en leyenda
    r2 = r2_score(x, y)
    # Texto debajo del gráfico
    axs.text(0.5, -0.15, f"$R^2$ para t+1: {r2:.3f}",
            transform=axs.transAxes,
            ha='center', va='top', fontsize=12)

    axs.set_xlabel("Valor real")
    axs.set_ylabel("Predicción")
    axs.legend()
    axs.set_title(f'Regresión para horizonte t+1')
    axs.grid(True)
    plt.tight_layout()    
    plt.savefig("SARIMAX/Reg_lines_SARIMAX.png", dpi=300)
    plt.close()
    modelo.save("SARIMAX/modelo_SARIMAX.pkl")
    with open("SARIMAX/Resultados.txt", "w", encoding="utf-8") as f:
        f.write(mensaje + "\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
    return modelo, mensaje, predicciones,mae, rmse, mse


def fit_sarimax_with_auto_arima(y, X, seasonal, m, verbose):
    """
    Ajusta un modelo SARIMAX usando los mejores parámetros encontrados por auto_arima.

    Args:
        y (DF): Serie temporal objetivo.
        X (DF): Variables exógenas.
        seasonal (bool): Si se debe buscar estacionalidad.
        m (int): Frecuencia estacional.
        verbose (bool): Si mostrar o no el progreso de auto_arima.

    Returns:
        model_fit (SARIMAXResultsWrapper): Modelo ajustado de SARIMAX.
        best_params (dict): Diccionario con los mejores parámetros encontrados.
    """
    
    # Buscar los mejores parámetros con auto_arima
    auto_model = auto_arima(
        y,
        exogenous=X,
        seasonal=seasonal,
        m=m,
        trace=verbose,
        start_p=0, max_p=1,
        start_q=0, max_q=1,
        start_P=0, max_P=1,
        start_Q=0, max_Q=0,
        max_order=5,       # p+q+P+Q ≤ 5
        stepwise=True,
        suppress_warnings=True,
        error_action='trace'
    )

    # Extraer los parámetros óptimos
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)

    if verbose:
       mensaje=f"Mejores parámetros encontrados: order={order}, seasonal_order={seasonal_order}"

    # Ajustar modelo SARIMAX con esos parámetros
    sarimax_model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    model_fit = sarimax_model.fit(disp=False)

    # Retornar modelo entrenado y parámetros usados
    return model_fit, mensaje


