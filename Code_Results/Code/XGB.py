from Preprocesado import input_length, escalar_dataset, crear_dataset_supervisado
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys

def crear_XGB(df, tr_size, vl_size, col_ref,lags, output_length, early_stopping,trees,depth,learning_rate,subsample,colsample,steps):
    train, val, test = input_length(df, tr_size, vl_size)
    print("Se acabo division")
    train_scaled, val_scaled, test_scaled, scaler = escalar_dataset(train, val, test, col_ref)
    print("Se acabo escalado")
    x_tr, y_tr = crear_dataset_supervisado(train_scaled, lags, output_length)
    x_vl, y_vl = crear_dataset_supervisado(val_scaled, lags, output_length)
    x_ts, y_ts = crear_dataset_supervisado(test_scaled, lags, output_length)
    print("Se acabo windowed")
    modelo_XGBOOST(df,x_tr,y_tr,x_vl,y_vl,x_ts,y_ts,scaler,early_stopping,trees,depth,learning_rate,subsample,colsample,steps,output_length)
    print("Se acabo modelado")

def modelo_XGBOOST(df,x_tr,y_tr,x_vl,y_vl,x_ts,y_ts,scaler,early_stopping,trees,depth,learning_rate,subsample,colsample,steps,output_length):

    X_train = x_tr.reshape((x_tr.shape[0], -1))  
    X_val = x_vl.reshape((x_vl.shape[0], -1)) 
    X_test= x_ts.reshape((x_ts.shape[0], -1))  
    y_train = y_tr.reshape((y_tr.shape[0], -1))
    y_val = y_vl.reshape((y_vl.shape[0], -1))
    y_test=y_ts.reshape((y_ts.shape[0], -1))  
    column_names = []
    for t in range(24):
        for col in df.columns:
            column_names.append(f"{col}-{t+1}")
    modelos = entrenar_multioutput_xgboost(
    X_train, y_train, X_val, y_val,column_names,
    early_stopping_rounds=early_stopping,
    n_estimators=trees,
    max_depth=depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample
    )
    os.makedirs("XGBOOST", exist_ok=True)
    metricas=["rmse","mae"]
    
    
    for metric in metricas:
        fig, axs = plt.subplots(y_train.shape[1],1,figsize=(8,5))
        for i,ax in enumerate (axs):
            evals_result = modelos[i].evals_result()
            result = evals_result['validation_0'][f'{metric}']
            result2 = evals_result['validation_1'][f'{metric}']
            ax.plot(result, label=f'{metric} (Validation)t+{i+1}')
            ax.plot(result2, label=f'{metric} (Train) t+{i+1}')
            ax.set_xlabel('Boosting Round')
            ax.set_ylabel(f'{metric}')
            ax.set_title(f'{metric} por iteración')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"XGBOOST/{metric}_XGB.png", dpi=300)
        plt.close()

    # Para cada modelo, predecimos sobre esas muestras
    
    predicciones = np.column_stack([model.predict(X_test[:steps]) for model in modelos])

    plt.plot(scaler.inverse_transform(predicciones[1:, [0]]), label='Predicciónt+1')
    plt.plot(scaler.inverse_transform(predicciones[:, [1]]), label='Predicciónt+2')
    plt.plot(scaler.inverse_transform(y_test[1:steps, [0]]), label='Real')
    plt.title(f'Predicción vs Real')
    plt.xlabel('Horizonte (horas)')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.savefig("XGBOOST/Ejemplo_XGB.png", dpi=300)
    plt.close()

    modelos[0].get_booster().feature_names = column_names
    # Plot feature importance
    fig,ax = plt.subplots(figsize=(10, 30))
    xgb.plot_importance(modelos[0], ax=ax)
    plt.savefig("XGBOOST/Feature_Imp_XGB.png", dpi=300)
    plt.close()

    pred = scaler.inverse_transform(np.column_stack([model.predict(X_test) for model in modelos]))
    y_real = scaler.inverse_transform(y_test)

    fig, axs = plt.subplots(y_test.shape[1],1,figsize=(12,10))
    for i,ax in enumerate(axs):
        x=y_real[:, i]
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
    plt.savefig("XGBOOST/Reg_lines_XGB.png", dpi=300)

    for i in range (y_train.shape[1]):
        modelos[i].save_model(f"XGBOOST/modelo_t{i+1}_XGB.json")


def entrenar_multioutput_xgboost(X_train, y_train, X_val, y_val,column_names, **xgb_params,):
    modelos = []
    n_outputs = y_train.shape[1]

    for i in range(n_outputs):
        model = XGBRegressor(
            eval_metric=['rmse', 'mae'],
            feature_names=column_names,
            **xgb_params,
        )
        model.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
            verbose=False
        )
        modelos.append(model)

    return modelos


if __name__ == "__main__":
    # Leer argumentos desde la línea de comandos
    tr_size = float(sys.argv[1])
    vl_size = float(sys.argv[2])
    col_ref = sys.argv[3]
    lags = int(sys.argv[4])
    output_length = int(sys.argv[5])
    early_stopping = int(sys.argv[6])
    trees = int(sys.argv[7])
    depth = int(sys.argv[8])
    learning_rate = float(sys.argv[9])
    subsample= float(sys.argv[10])
    colsample= float(sys.argv[11])
    steps = int(sys.argv[12])
    time_col = sys.argv[13]
    path_csv = sys.argv[14]
    file_path = os.path.join("dataset", path_csv)
    df=pd.read_csv(file_path, index_col=time_col, parse_dates=True)
    print("Se acabo envio")
    crear_XGB(df, tr_size, vl_size, col_ref, lags, output_length, early_stopping,trees,depth,learning_rate,subsample,colsample,steps)