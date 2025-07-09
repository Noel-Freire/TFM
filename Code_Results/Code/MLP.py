from Preprocesado import input_length, escalar_dataset, crear_dataset_supervisado,predecir
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def crear_MLP(df, tr_size, vl_size, col_ref,lags, output_length, epochs, batch_size, num_layers, units, Do, steps):
    train, val, test = input_length(df, tr_size, vl_size)
    print("Se acabo division")
    train_scaled, val_scaled, test_scaled, scaler = escalar_dataset(train, val, test, col_ref)
    print("Se acabo escalado")
    x_train, y_train = crear_dataset_supervisado(train_scaled, lags, output_length)
    x_val, y_val = crear_dataset_supervisado(val_scaled, lags, output_length)
    x_test, y_test = crear_dataset_supervisado(test_scaled, lags, output_length)
    print("Se acabo windowed")
    modelo_MLP(x_train, y_train, x_val, y_val, x_test, y_test, scaler, num_layers, units, epochs, batch_size, Do)
    print("Se acabo modelado")
    
def modelo_MLP(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, scaler, num_layers, units, epochs, batch_size, Do):
    input_shape = (x_tr.shape[1]*x_tr.shape[2],)  # input plano (lags × variables)
    output_length = y_tr.shape[1]  # ajustar según tu problema
    model = Sequential()
    # Añadir la primera capa (requiere input_shape)
    model.add(Dense(units[0], activation='relu',input_shape=input_shape))
    model.add(Dropout(Do))  # opcional, ayuda contra overfitting
    # Capas intermedias
    for  i,neurons in zip(range(1, num_layers), units[1:]):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(Do))
    # Capa de salida
    model.add(Dense(output_length))

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
        # Compilar
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()],
    )
    x_tr = x_tr.reshape((x_tr.shape[0], x_tr.shape[1] * x_tr.shape[2]))
    x_vl = x_vl.reshape((x_vl.shape[0], x_vl.shape[1] * x_vl.shape[2]))
    x_ts = x_ts.reshape((x_ts.shape[0], x_ts.shape[1] * x_ts.shape[2]))
    y_tr = y_tr.reshape((y_tr.shape[0], y_tr.shape[1]))
    y_vl = y_vl.reshape((y_vl.shape[0], y_vl.shape[1]))
    y_ts = y_ts.reshape((y_ts.shape[0], y_ts.shape[1]))
    # Entrenamiento (aproximadamente 1 min usando GPU)

    historia = model.fit(
        x = x_tr,
        y = y_tr,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (x_vl, y_vl),
        callbacks=[early_stop],
        verbose=2
    )
    os.makedirs("MLP", exist_ok=True)
    plt.plot(historia.history['mse'],label='MSE train')
    plt.plot(historia.history['val_mse'],label='MSE val')
    plt.xlabel('Iteración')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig("MLP/MSE_MLP.png")
    plt.close()

    plt.plot(historia.history['mae'],label='MAE train')
    plt.plot(historia.history['val_mae'],label='MAE val')
    plt.xlabel('Iteración')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("MLP/MAE_MLP.png")
    plt.close()

    plt.plot(historia.history['root_mean_squared_error'],label='RMSE train')
    plt.plot(historia.history['val_root_mean_squared_error'],label='RMSE val')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig("MLP/RMSE_MLP.png")
    plt.close()
     #RMSE bars
    rmse_list = []
    y_real=scaler.inverse_transform(y_ts.reshape(-1, output_length))
    pred=predecir(x_ts, model, scaler)
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
    plt.savefig("MLP/RMSE_bar_MLP.png", dpi=300)
    plt.close()

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
    plt.savefig("MLP/Reg_lines_MLP.png", dpi=300)
    plt.close()
    
    fig3, ax = plt.subplots()
    for i in range (0,output_length):
        ax.plot(pred[output_length-i:steps+output_length,i],label=f'predicción en t+{i+1}')

    ax.plot(y_real[output_length:steps+output_length,0].reshape(-1),label='Valor real')
    ax.set_xlabel('Horas')
    ax.set_ylabel('Potencia')
    ax.legend()
    plt.savefig("MLP/Ejemplo_MLP.png", dpi=300, bbox_inches='tight')
    plt.close()  # Limpia la figura
    model.save('MLP/modelo_MLP.keras')


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
    Do = float(sys.argv[10])
    steps = int(sys.argv[11])
    time_col = sys.argv[12]
    path_csv = sys.argv[13]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    print("Se acabo envio")
    crear_MLP(df, tr_size, vl_size, col_ref, lags, output_length, epochs, batch_size, num_layers, units, Do,steps)


