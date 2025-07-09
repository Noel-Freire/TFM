import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import sys
import os


def create_elbow_plot(datos, ref, lags, n_clusters, amount):
    a = 0
    cluster = []
    for i in range(0, datos.shape[0] - lags, lags):
        cluster.append(datos[ref].values[i:i+lags])
        a += 1
    Sum_of_squared_distances = []
    K= range (2,n_clusters+1)
    for k in K:
        km = TimeSeriesKMeans(n_clusters=k,
                            n_init=2,
                            metric="dtw",
                            verbose=False,
                            max_iter_barycenter=10,
                            random_state=0,
                            n_jobs=-1)

        km = km.fit(cluster[:amount])
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('elbow_plot.png')


if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    col_ref = sys.argv[1]
    lags = int(sys.argv[2])
    n_clusters_max = int(sys.argv[3])
    amount = int(sys.argv[4])
    path_csv = sys.argv[5]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=None, parse_dates=True)
    print("Se acabo envio")
    outliers=create_elbow_plot(df, col_ref, lags, n_clusters_max, amount)