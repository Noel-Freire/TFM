import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from Preprocesado import escalar_dataset

def create_windows(series, window_size):
    X = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
    return np.array(X)


def autoencoder(df, col_ref, lags, percentile):
    df = df.reset_index(drop=True)
    df2, _, _, _ = escalar_dataset(df, df, df, col_ref)
    # Autoencoder simple (input = output)
    autoencoder = Sequential([
        Dense(8, activation='relu', input_shape=(lags,)),
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(lags, activation='sigmoid')
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    df3 = pd.DataFrame(df2, columns=list(df.columns.values))
    X_windows = create_windows(df3[col_ref], lags)
    # Entrenar el autoencoder
    autoencoder.fit(X_windows, X_windows, epochs=50, batch_size=16, verbose=0)
    # Reconstrucción y error
    X_pred = autoencoder.predict(X_windows)
    mse = np.mean(np.power(X_windows - X_pred, 2), axis=1)
    # Umbral para detectar outliers (p.ej. 95 percentil)
    threshold = np.percentile(mse, percentile)
    outliers = np.where(mse > threshold)[0]
    # Visualización
    plt.plot(df[col_ref].values, label='Serie temporal (train)')
    outlier_indices = outliers + lags -1
    outlier_indices = outlier_indices[outlier_indices < len(df[col_ref])]
    plt.scatter(outlier_indices, df[col_ref].values[outlier_indices], color='red', label='Outliers detectados')
    plt.legend()
    plt.savefig('outliers_autoencoder.png')
    return outlier_indices


if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    col_ref = sys.argv[1]
    lags = int(sys.argv[2])
    percentile = float(sys.argv[3])
    time_col = sys.argv[4]
    path_csv = sys.argv[5]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    outliers = autoencoder(df, col_ref, lags, percentile)
    np.set_printoptions(threshold=np.inf)
    print(outliers)