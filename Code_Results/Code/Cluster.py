import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import sys
import os

def create_cluster_plot(datos, col_ref, lags, n_clusters_exacto, amount):
    cluster = []
    for i in range(0, datos.shape[0] - lags, lags):
        cluster.append(datos[col_ref].values[i:i+lags]) 

    km = TimeSeriesKMeans(n_clusters=n_clusters_exacto,
                        n_init=2,
                        metric="dtw",
                        verbose=False,
                        max_iter_barycenter=10,
                        random_state=0,
                        n_jobs=-1)
    km.fit(cluster[:amount])
    # Obtener los centroides (valores medios de cada clase)
    centroides = km.cluster_centers_
    plt.figure(figsize=(12, 6))
    for i, centroide in enumerate(centroides):
        plt.plot(centroide.ravel(), label=f'Cluster {i}')
    plt.title(f"Centroides de los {n_clusters_exacto} clusters (k={n_clusters_exacto})")
    plt.legend()
    plt.savefig(f"centroides.png")


if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    col_ref = sys.argv[1]
    lags = int(sys.argv[2])
    n_clusters_exacto= int(sys.argv[3])
    amount = int(sys.argv[4])
    path_csv = sys.argv[5]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=None, parse_dates=True)
    print("Se acabo envio")
    create_cluster_plot(df, col_ref, lags, n_clusters_exacto, amount)