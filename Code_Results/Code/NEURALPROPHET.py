from Preprocesado import input_length
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

def crear_NEURALPROPHET(datos, tr_size, vl_size, col_ref, col_time, format_date, freq, lags, output_length, epochs, learning_rate, yearly, weekly, daily, steps):
    print(datos[col_time])
    datos[col_time] = pd.to_datetime(datos[col_time], format='%Y-%m-%d %H:%M')
    datos['y'] = datos[col_ref].shift(-output_length)
    datos['ds'] = datos[col_time]
    del datos[col_time]
    datos = datos.interpolate()

    train, val, test = input_length(datos, tr_size, vl_size)
   # Dividir en conjuntos de entrenamiento, validación y prueba

   # Crear y ajustar el modelo
    modelo = NeuralProphet(
    yearly_seasonality=yearly,
    weekly_seasonality=weekly,
    daily_seasonality=daily,
    learning_rate=learning_rate,
    n_forecasts=output_length,
    n_lags=lags
    )
    regresores = [col for col in datos.columns if col not in ['y', 'ds']]
    for reg in regresores:
        modelo.add_lagged_regressor(reg, n_lags=lags)
    metrics = modelo.fit(train, freq=freq,epochs=epochs, validation_df=val,early_stopping=True,learning_rate=learning_rate)

    os.makedirs('NEURALPROPHET', exist_ok=True)
    plt.plot(metrics['RMSE'],label='RMSE train')
    plt.plot(metrics['RMSE_val'],label='RMSE val')
    plt.xlabel('Iteración')
    plt.ylabel('Watts')
    plt.legend();
    plt.savefig('NEURALPROPHET/RMSE_NEURALPROPHET.png')
    plt.close()

    plt.plot(metrics['Loss'],label='Smooth L1 Loss (MAE+MSE)')
    plt.plot(metrics['Loss_val'],label='Smooth L1 Loss val')
    plt.xlabel('Iteración')
    plt.ylabel('Smooth L1 Loss (MAE+MSE)')
    plt.legend();
    plt.savefig('NEURALPROPHET/LOSS_NEURALPROPHET.png')
    plt.close()

    plt.plot(metrics['MAE'],label='MAE train')
    plt.plot(metrics['MAE_val'],label='MAE val')
    plt.xlabel('Iteración')
    plt.ylabel('MAE')
    plt.legend();
    plt.savefig('NEURALPROPHET/MAE_NEURALPROPHET.png')
    plt.close()

    predicciones = []
    fechas = []

    for i in range(steps):
        current_lag = test.iloc[i : i + lags].copy()
        
        if len(current_lag) < lags:
            break  # no hay suficiente data al final
        # Crear el dataframe futuro 
        future_df = modelo.make_future_dataframe(current_lag, n_historic_predictions=False)

        # Predecir solo 1 paso
        pred = modelo.predict(future_df)

        # Guardar predicción y fecha
        fila=[]
        for h in range(1, output_length + 1):
            valor = pred[f"yhat{h}"].values[lags + h-1]
            fila.append(valor)
        fechas.append(pred["ds"].values[0])
        predicciones.append(fila)
        # Crear dataframe con resultados
    predicciones_np = np.array(predicciones)

    # Construir el diccionario de resultados
    result_dict = {
        "ds": fechas,
        "y_real": test["y"].iloc[lags : lags + len(predicciones)].values
    }
    for i in range(1, output_length + 1):
        result_dict[f"yhat{i}"] = predicciones_np[:, i-1]

    df_pred = pd.DataFrame(result_dict)
    plt.figure(figsize=(12, 5))
    plt.plot(df_pred["ds"], df_pred["y_real"], label="Real")
    for i in range(1, output_length + 1):
        desplazado = df_pred["ds"] + pd.to_timedelta(-i+1, unit="h")  # o "D" según tu frecuencia
        plt.plot(desplazado, df_pred[f"yhat{i}"], label=f"Predicción t+{i}")
    plt.legend()
    plt.title("Predicción rolling (ventana deslizante)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('NEURALPROPHET/Ejemplo_NEURALPROPHET.png')
    plt.close()
    # Refresion_lines 
    for j in range (1, lags + 1):
        for i in range(20):
            current_lag = test.iloc[i : i + lags].copy()
            if len(current_lag) < lags:
                break  
            future_df = modelo.make_future_dataframe(current_lag, n_historic_predictions=False)
            pred = modelo.predict(future_df)
            fila=[]
            for h in range(1, output_length + 1):
                valor = pred[f"yhat{h}"].values[lags + h-1]
                fila.append(valor)
            fechas.append(pred["ds"].values[0])
            predicciones.append(fila)
    fig, axs = plt.subplots(output_length,1,figsize=(12,10))
    predicciones_np = np.array(predicciones)
    for i,ax in enumerate(axs):
        x = test["y"][lags+i+1:predicciones_np.shape[0]+lags+1+i]
        y = predicciones_np[:, i]

        # Ajuste lineal: coeficientes [pendiente, intercepto]
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)

        # Dibujar puntos reales vs predichos
        ax.scatter(x, y, label=f'Datos t+{i+1}', alpha=0.5)

        # Dibujar recta de regresión
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, poly1d_fn(x_line), label=f'Recta t+{i+1}',color='red')

        # R2 para info en leyenda
        r2 = r2_score(x, y)
            # Texto debajo del gráfico
        ax.text(0.5, -0.65, f"$R^2$ para t+{i+1}: {r2:.3f}",
                transform=ax.transAxes,
                ha='center', va='top', fontsize=12)

        ax.set_xlabel("Valor real")
        ax.set_ylabel("Predicción")
        ax.legend()
        ax.set_title(f'Regresión para horizonte t+{i+1}')
        ax.grid(True)

    plt.tight_layout()    
    plt.savefig('NEURALPROPHET/Regresion_lines_NEURALPROPHET.png')
    plt.close()

if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    tr_size = float(sys.argv[1])
    vl_size = float(sys.argv[2])
    col_ref = sys.argv[3]
    lags = int(sys.argv[4])
    output_length = int(sys.argv[5])
    epochs = int(sys.argv[6])
    col_time = sys.argv[7]
    format_date = sys.argv[8]
    freq = sys.argv[9]
    learning_rate = float(sys.argv[10])
    yearly = bool(sys.argv[11])
    weekly = bool(sys.argv[12])
    daily = bool(sys.argv[13])
    steps = int(sys.argv[14])
    path_csv = sys.argv[15]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, parse_dates=True)
    print("Se acabo envio")  
    crear_NEURALPROPHET(df, tr_size, vl_size, col_ref, col_time, format_date, freq, lags, output_length, epochs, learning_rate, yearly, weekly, daily, steps)
    
