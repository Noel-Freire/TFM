from sklearn.ensemble import IsolationForest
from Preprocesado import escalar_dataset
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np


def create_windows(data, size):
    X = []
    for i in range(len(data) - size+1):
        X.append(data[i:i+size])
    return np.array(X)

def i_forest(df,col_ref,lags,contamination):
    df = df.reset_index(drop=True)
    df2, _, _, scaler = escalar_dataset(df, df, df, col_ref)
    df3 = pd.DataFrame(df2, columns=list(df.columns.values))
    X = create_windows(df3[col_ref], lags)
    P_windows = X
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(P_windows)
    preds = model.predict(P_windows)  # -1 = outlier, 1 = normal
    outliers = np.where(preds == -1)[0]
    outlier_indices = outliers + lags - 1
    outlier_indices = outlier_indices[outlier_indices < len(df)]
    # Visualización
    plt.plot(df[col_ref].values, label='Serie temporal')
    plt.scatter(outlier_indices, df[col_ref].values[outlier_indices], color='red', label='Outliers detectados')
    plt.legend()
    plt.savefig('outliers_isolation_forest.png')
    return outlier_indices

if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    col_ref = sys.argv[1]
    lags = int(sys.argv[2])
    contamination = float(sys.argv[3])
    time_col = sys.argv[4]
    path_csv = sys.argv[5]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    outliers = i_forest(df, col_ref, lags, contamination)
    np.set_printoptions(threshold=np.inf)
    print(outliers)