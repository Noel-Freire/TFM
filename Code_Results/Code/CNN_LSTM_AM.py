from Preprocesado import input_length, escalar_dataset, crear_dataset_supervisado,predecir
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, Layer,MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def crear_CNN_LSTM_AM(df, tr_size, vl_size, col_ref,lags, output_length, num_layers_Conv, num_layers_LSTM, units, filtros, Do, batch_size, epoch,steps):
    train, val, test = input_length(df, tr_size, vl_size)
    print("Se acabo division")
    train_scaled, val_scaled, test_scaled, scaler = escalar_dataset(train, val, test, col_ref)
    print("Se acabo escalado")
    x_train, y_train = crear_dataset_supervisado(train_scaled, lags, output_length)
    x_val, y_val = crear_dataset_supervisado(val_scaled, lags, output_length)
    x_test, y_test = crear_dataset_supervisado(test_scaled, lags, output_length)
    print("Se acabo windowed")
    modelo_CNN_LSTM_AM(x_train, y_train, x_val, y_val, x_test, y_test, scaler, num_layers_Conv, num_layers_LSTM, units, filtros, Do, batch_size, epoch)
    print("Se acabo modelado")

def modelo_CNN_LSTM_AM(x_tr, y_tr, x_vl, y_vl, x_ts, y_ts,scaler, num_layers_Conv, num_layers_LSTM, units, filtros, Do, batch_size, epoch):
    input_shape=(x_tr.shape[1], x_tr.shape[2])
    output_length=y_tr.shape[1]
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=filtros[0], kernel_size=3, padding='same', activation='relu')(inputs)
    x = Dropout(Do)(x)
    # Añadir capas intermedias (si las hay)
    for i,filtro in zip(range(1, num_layers_Conv), filtros[1:]):
        x = Conv1D(filters=filtro, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(Do)(x)
    x = MaxPooling1D(pool_size=2)(x)
    

    # Añadir capas intermedias (si las hay)
    for i,neurons in zip(range(0, num_layers_LSTM), units):
        x = LSTM(neurons, return_sequences=True)(x)
        x = Dropout(Do)(x)

    attention = Attention()(x)
    outputs = Dense(output_length)(attention)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    
    early_stop = EarlyStopping(
    monitor='val_loss',      # Métrica a monitorear
    patience=10,             # Número de épocas sin mejora para detener
    restore_best_weights=True  # Restaura los pesos de la mejor época
    )

    historia = model.fit(
        x = x_tr,
        y = y_tr,
        batch_size = batch_size,
        epochs = epoch,
        validation_data = (x_vl, y_vl),
        callbacks=[early_stop],
        verbose=2
    )
    os.makedirs("CNN_LSTM_AM", exist_ok=True)
    plt.plot(historia.history['loss'],label='MSE train')
    plt.plot(historia.history['val_loss'],label='MSE val')
    plt.xlabel('Iteración')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig("CNN_LSTM_AM/MSE_CNN_LSTM_AM.png", dpi=300)
    plt.close()

    plt.plot(historia.history['mean_absolute_error'],label='MAE train')
    plt.plot(historia.history['val_mean_absolute_error'],label='MAE val')
    plt.xlabel('Iteración')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig("CNN_LSTM_AM/MAE_CNN_LSTM_AM.png", dpi=300)
    plt.close()

    plt.plot(historia.history['root_mean_squared_error'],label='RMSE train')
    plt.plot(historia.history['val_root_mean_squared_error'],label='RMSE val')
    plt.xlabel('Iteración')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig("CNN_LSTM_AM/RMSE_CNN_LSTM_AM.png", dpi=300)
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
    plt.savefig("CNN_LSTM_AM/RMSE_bar_CNN_LSTM_AM.png", dpi=300)
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
    plt.savefig("CNN_LSTM_AM/Reg_lines_CNN_LSTM_AM.png", dpi=300)

    fig3, ax = plt.subplots()
    for i in range (0,output_length):
        ax.plot(pred[output_length-i:steps+output_length,i],label=f'predicción en t+{i+1}')

    ax.plot(y_real[output_length:steps+output_length,0].reshape(-1),label='Valor real')
    ax.set_xlabel('Horas')
    ax.set_ylabel('Potencia')
    ax.legend()
    plt.savefig("CNN_LSTM_AM/Ejemplo_CNN_LSTM_AM.png", dpi=300, bbox_inches='tight')
    plt.close()  # Limpia la figura
    model.save('CNN_LSTM_AM/modelo_CNN_LSTM_AM.keras')


if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    tr_size = float(sys.argv[1])
    vl_size = float(sys.argv[2])
    col_ref = sys.argv[3]
    lags = int(sys.argv[4])
    output_length = int(sys.argv[5])
    num_layers_Conv = int(sys.argv[6])
    num_layers_LSTM = int(sys.argv[7])
    units = sys.argv[8]
    units = [int(x) for x in units.split(",")]
    filtros = sys.argv[9]
    filtros = [int(x) for x in filtros.split(",")]
    Do = float(sys.argv[10])
    batch_size = int(sys.argv[11])
    epoch = int(sys.argv[12])
    steps = int(sys.argv[13])
    time_col = sys.argv[14]
    path_csv = sys.argv[15]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    print("Se acabo envio")
    crear_CNN_LSTM_AM(df, tr_size, vl_size, col_ref, lags, output_length, num_layers_Conv, num_layers_LSTM, units, filtros, Do, batch_size, epoch, steps)