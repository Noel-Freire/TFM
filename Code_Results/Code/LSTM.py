from Preprocesado import input_length, escalar_dataset, crear_dataset_supervisado,predecir
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import os


def crear_LSTM(df, tr_size, vl_size, col_ref,lags, output_length, epochs, batch_size, num_layers, units,Do,steps):
    train, val, test = input_length(df, tr_size, vl_size)
    print("Se acabo division")
    train_scaled, val_scaled, test_scaled, scaler = escalar_dataset(train, val, test, col_ref)
    print("Se acabo escalado")
    x_train, y_train = crear_dataset_supervisado(train_scaled, lags, output_length)
    x_val, y_val = crear_dataset_supervisado(val_scaled, lags, output_length)
    x_test, y_test = crear_dataset_supervisado(test_scaled, lags, output_length)
    print("Se acabo windowed")
    modelo_LSTM(x_train,y_train, x_val, y_val,x_test, y_test, scaler, num_layers, units, epochs, batch_size,Do,steps)
    print("Se acabo modelado")
    




def modelo_LSTM(x_tr,y_tr, x_vl, y_vl,x_ts, y_ts, scaler, num_layers, units, epochs, batch_size,Do,steps):
    output_length = y_tr.shape[1]
    # Entrada del modelo
    input_shape = (x_tr.shape[1], x_tr.shape[2])

    # Crear modelo secuencial
    model = Sequential()
    # Añadir la primera capa LSTM (requiere input_shape)
    model.add(LSTM(units[0], return_sequences=(num_layers > 1), input_shape=input_shape))
    # Añadir capas intermedias que desee el usuario (si las hay)
    for i,neurons in zip(range(1, num_layers - 1), units[1:-1]):
        model.add(LSTM(neurons, return_sequences=True))
        model.add(Dropout(Do))
    # Añadir la última capa LSTM (sin return_sequences)
    if num_layers > 1:
        model.add(LSTM(units[-1]))

    # Capa de salida
    model.add(Dense(output_length))  # Desnse para la salida de los valores predichos

    # Compilar
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=1e-4),  # Optimizer Adam con learning rate = 1e-4
        metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()]
    )

    early_stop = EarlyStopping(
    monitor='val_loss',     # Lo que quieres vigilar (puede ser 'val_mae', 'val_mse', etc.)
    patience=5,            # Nº de epochs sin mejora antes de parar
    restore_best_weights=True  # Recupera los mejores pesos al final
)
    # Entrenamiento
    historia = model.fit(
        x = x_tr,
        y = y_tr,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (x_vl, y_vl),
        callbacks=[early_stop],
        verbose=2,
    )
    # Figura MAE
    fig1, ax = plt.subplots()
    ax.plot(historia.history['mae'], label='MAE train')
    ax.plot(historia.history['val_mae'], label='MAE val')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('MAE')
    ax.legend()
    os.makedirs("LSTM", exist_ok=True)
    plt.savefig("LSTM/MAE_LSTM.png", dpi=300, bbox_inches='tight')
    plt.close()  # Limpia la figura
    #Figura MSE
    fig1, ax = plt.subplots()
    ax.plot(historia.history['mse'], label='MSE train')
    ax.plot(historia.history['val_mse'], label='MSE val')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.savefig("LSTM/MSE_LSTM.png", dpi=300, bbox_inches='tight')
    plt.close()  # Limpia la figura

    plt.plot(historia.history['root_mean_squared_error'],label='RMSE train')
    plt.plot(historia.history['val_root_mean_squared_error'],label='RMSE val')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE')
    plt.legend();
    plt.savefig("LSTM/RMSE_LSTM.png", dpi=300)
    plt.close()

    pred=predecir(x_ts, model, scaler)
    fig3, ax = plt.subplots()
    for i in range (0,output_length):
        ax.plot(pred[output_length-i:steps+output_length,i],label=f'predicción en t+{i+1}')

    ax.plot(scaler.inverse_transform(y_ts[output_length:steps+output_length,0]).reshape(-1),label='Valor real')
    ax.set_xlabel('Horas')
    ax.set_ylabel('Potencia')
    ax.legend();
    plt.savefig("LSTM/Ejemplo_LSTM.png", dpi=300, bbox_inches='tight')
    plt.close()  # Limpia la figura

    y_real=scaler.inverse_transform(y_ts.reshape(-1, output_length))

    fig, axs = plt.subplots(output_length,1,figsize=(12,10))

    for i,ax in enumerate(axs):
        x = y_real[:, i]
        y = pred[:, i]

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
    plt.savefig("LSTM/Reg_lines_LSTM.png", dpi=300)
    plt.close()

    rmse_list = []
    for i in range(output_length):
        rmse = np.sqrt(mean_squared_error(y_real[:, i], pred[:, i]))
        rmse_list.append(rmse)
    horizontes = [f't+{i+1}' for i in range(output_length)]
    # Graficar barras
    plt.figure(figsize=(8, 5))
    plt.bar(horizontes, rmse_list)
    plt.xlabel('Horizonte de predicción')
    plt.ylabel('RMSE (W)')
    plt.title('Error RMSE por horizonte de predicción (LSTM)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Mostrar los valores numéricos arriba de las barras
    for i, val in enumerate(rmse_list):
        plt.text(i, val + 0.01 * max(rmse_list), f'{val:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("LSTM/RMSE_bar_LSTM.png", dpi=300)
    plt.close()

    model.save('LSTM/modelo_LSTM.keras')

if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    tr_size = float(sys.argv[1])
    vl_size = float(sys.argv[2])
    col_ref = sys.argv[3]
    lags = int(sys.argv[4])
    output_length = int(sys.argv[5])
    epochs = int(sys.argv[6])
    batch_size = int(sys.argv[7])
    num_layers = int(sys.argv[8])
    units = sys.argv[9]
    units = [int(x) for x in units.split(",")]
    Do=float(sys.argv[10])
    steps = int(sys.argv[11])
    time_col = sys.argv[12]
    path_csv = sys.argv[13]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    print("Se acabo envio")
    crear_LSTM(df, tr_size, vl_size, col_ref, lags, output_length, epochs, batch_size, num_layers, units, Do,steps)
