import os
import streamlit as st
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
import base64
import plotly.express as px
import plotly.io as pio
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
from SARIMAX import crear_SARIMAX
from streamlit_option_menu import option_menu
from datetime import datetime
from PIL import Image
import requests
import subprocess
import shutil
import re
import numpy as np
from numpy import sqrt, log
from Textos import texto_intro1,texto_intro2,texto_test,texto_frecuencia
from Operaciones_datos import frecuencia_usuario
from Graficas_series_temp import plot_acf_pacf, fft,test_adfuller,descompose_seasonal,seasonal_SLT,test_kpss


import io
from matplotlib.figure import Figure as MplFigure
from PIL import Image
import plotly.graph_objs as go
import plotly.io as pio


# Inicializar claves de session_state 
if 'frecuencia_confirmada' not in st.session_state:
    st.session_state.frecuencia_confirmada = False
if 'columnas_confirmadas' not in st.session_state:
    st.session_state.columnas_confirmadas = False
if 'preprocesado_realizado' not in st.session_state:
    st.session_state.preprocesado_realizado = False
if 'imagen_acf_pacf' not in st.session_state:
    st.session_state.imagen_acf_pacf=None
if 'STL' not in st.session_state:
    st.session_state.STL=None
if 'imagen_Seasonal_Decompose' not in st.session_state:
    st.session_state.imagen_Seasonal_Decompose=None
if "adf_result" not in st.session_state:
    st.session_state.adf_result = None
if "kpss_result" not in st.session_state:
    st.session_state.kpss_result = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'time_col' not in st.session_state:
    st.session_state.time_col = None
if 'format_date' not in st.session_state:
    st.session_state.format_date = None
if "rutas_zip" not in st.session_state:
    st.session_state["rutas_zip"] = {}
if "cant" not in st.session_state:
    st.session_state.cant = None
if "pacf" not in st.session_state:
    st.session_state.pacf = []
if "acf" not in st.session_state:
    st.session_state.acf = []


st.set_page_config(
    page_title="ML App",
    layout="wide",
)


def embed_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
    return pdf_display

#st.write('Musica de fondo para programar')
#st.audio('A.mp3') 

selected = option_menu(
    menu_title="Bienvenido al tratamiento de Series Temporales",
    options=["Inicio","Forecasting", "Clustering", "Datasets de ejemplo", "Más información"],
    icons=["boxes",  "graph-up-arrow","exclude","filetype-csv", "info",],
    menu_icon="stack",
    orientation="horizontal")

# Cargar el dataset si existe
if os.path.exists("dataset.csv"):
    df = pd.read_csv("dataset.csv", index_col=None)

if selected == "Inicio":
    body = texto_intro1()
    la1, la2 = st.columns(2)
    with la1:
        st.image("https://media.licdn.com/dms/image/v2/C4E1BAQHK47kpMotvuw/company-background_10000/company-background_10000/0/1590229205371/delegacin_eps_ferrol_cover?e=1752148800&v=beta&t=YHLm97lTTnqkkjb-q8B_PDoGoZPsZ4HfuzTrTgWYzWI")
    with la2:
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed #FF4B4B'>",
                unsafe_allow_html=True)
    aa1, aa2 = st.columns(2)
    with aa1:
        body = texto_intro2()
        st.markdown(body, unsafe_allow_html=True)
    with aa2:
            st.image("https://economiacolaborativaudcderecho.wordpress.com/wp-content/uploads/2018/02/logo-udc-e1518280684389.png",width=1000)

    with st.expander("📘 Documento"):
        st.write("Visualizando el documento:")
        pdf_html = embed_pdf("TSFE.pdf")  
        st.markdown(pdf_html, unsafe_allow_html=True)


if selected == "Forecasting":
    st.title("Series Temporales")
    # Sidebar
    with st.sidebar:
        st.image("https://www.udc.es/export/sites/udc/epef/.galleries/EPEF/LogoEPEF.jpg_2063069294.jpg")
        st.title("Forecasting")
        choice = st.radio("Selección", ["Subir datos", "Informacion General de todas las Covariables","Preprocesado de series temporales","Detector Outliers","Graficas Preprocesado","Solucionar la NO ESTACIONARIEDAD", "Modelado", "Descargar modelo"])
        st.info("Aplicación para la creación de modelos de Forecasting de series temporales.")
        st.info("Desarrollado por:")
        st.info("Noel Freire Mahía")

        # Cargar archivo
    if choice == "Subir datos":
        st.title("Cargar archivo local")
        # Listar archivos disponibles en la carpeta data
        files = [f for f in os.listdir("dataset") if f.endswith(('.csv', '.xlsx', '.json'))]

        if files:
            file_selected = st.selectbox("Selecciona un archivo en la carpeta 'dataset/'", files)
            
            if st.button("Cargar archivo"):
                st.session_state.clear()
                file_path = os.path.join("dataset", file_selected)
                file_type = file_selected.split('.')[-1]

                with st.spinner("Cargando el archivo..."):
                    if file_type == "csv":
                        df = pd.read_csv(file_path)
                    elif file_type in ["xlsx", "xls"]:
                        df = pd.read_excel(file_path)
                    elif file_type == "json":
                        df = pd.read_json(file_path)
                    else:
                        st.error("Tipo de archivo no soportado.")
                        df = None

                if 'df' in locals():
                    st.success("Archivo cargado exitosamente")
                    st.dataframe(df)
                    # Guardar si quieres reutilizar
                    df.to_csv("dataset.csv", index=False)
                    st.session_state.df=df
            else:
                st.warning("No hay archivos en la carpeta 'data/'.")
                st.info("Por favor, coloca un archivo grande en la carpeta 'data/' y actualiza la página.")
    # Perfilado con ydata-profiling
    if choice == "Informacion General de todas las Covariables":
        st.title("Explorar el conjunto de datos")
        if 'df' in locals():
            profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            html(profile.to_html(), height=1300, scrolling=True)
        else:
            st.warning("No se ha subido ningún dataset aún.")
            st.info("Por favor, sube un dataset para proceder con el preprocesado.")

    
    if choice == "Preprocesado de series temporales":
        st.title("Explorar el conjunto de datos")
        if 'df' in locals():
            # Paso 1: Seleccionar frecuencia
            st.markdown(texto_frecuencia())
            frecuencia = st.selectbox("Ingresar abreviatura de frecuencia:", ["S", "T","10min", "H", "D", "M", "Y"])

            if st.button("Frecuencia seleccionada"):
                st.session_state.frecuencia_confirmada = True
                st.session_state.frecuencia = frecuencia

            # Paso 2: Seleccionar columnas de tiempo y target
            if st.session_state.frecuencia_confirmada:
                time = st.selectbox("Elije la columna del tiempo", df.columns)
                chosen_target = st.selectbox("Elije la columna para predecir", df.columns)

                if st.button("Columnas seleccionadas"):
                    st.session_state.columnas_confirmadas = True
                    st.session_state.time_col = time
                    st.session_state.target_col = chosen_target

            # Paso 3: Formato de fecha y preprocesado
            if st.session_state.columnas_confirmadas:
                st.markdown("**Formato de fecha**")
                st.markdown("Escribe el formato de la fecha que tiene el dataset. Por ejemplo:")
                st.markdown("%Y-%m-%d %H:%M para 2023-10-01 12:00")
                st.markdown("%d-%b-%Y  %H:%M:%S para 01-Oct-2023 12:00:00")

                # Mostrar ejemplo real
                ejemplo = df[st.session_state.time_col].iloc[0]
                st.markdown(f"Ejemplo del dataset: **{ejemplo}**")

                format_date = st.text_input("Ingresar el formato de la fecha.")
                
                if st.button("Preprocesar"):
                    with st.spinner("Preprocesando el dataset..."):
                        df=frecuencia_usuario(df, st.session_state.time_col, st.session_state.frecuencia, format_date)
                        st.session_state.preprocesado_realizado = True
                        st.session_state.df = df.copy()
                        st.session_state.format_date = format_date
            # Mostrar resultado
            if st.session_state.preprocesado_realizado:
                df = st.session_state.df
                st.success("✅ Dataset preprocesado exitosamente")
                st.dataframe(st.session_state.df.head(n=70))
                nan_cols = df.columns[df.isna().any()].tolist()# Verificar si hay valores nulos
                st.session_state.nan_cols = nan_cols
                if nan_cols:
                    st.markdown("Estas columnas tienen valores nulos:")
                    st.write(df[nan_cols].isna().sum())
                    selected_col = st.selectbox("Selecciona una columna para tratar NaNs", nan_cols)
                    metodo = st.selectbox("Selecciona el método para tratar los NaNs", [
                        "Forward Fill (ffill)",
                        "Backward Fill (bfill)",
                        "Interpolación",
                        "Rellenar con 0",
                        "Rellenar con la media",
                        "Rellenar con la mediana",
                        "Eliminar filas con NaNs"
                    ])
                    if st.button("Aplicar tratamiento"):
                        if metodo == "Forward Fill (ffill)":
                            df[selected_col] = df[selected_col].fillna(method='ffill')
                        elif metodo == "Backward Fill (bfill)":
                            df[selected_col] = df[selected_col].fillna(method='bfill')
                        elif metodo == "Interpolación":
                            df[selected_col] = df[selected_col].interpolate()
                        elif metodo == "Rellenar con 0":
                            df[selected_col] = df[selected_col].fillna(0)
                        elif metodo == "Rellenar con la media":
                            media = df[selected_col].mean()
                            df[selected_col] = df[selected_col].fillna(media)
                        elif metodo == "Rellenar con la mediana":
                            mediana = df[selected_col].median()
                            df[selected_col] = df[selected_col].fillna(mediana)
                        elif metodo == "Eliminar filas con NaNs":
                            df = df.dropna(subset=[selected_col])
                        
                        st.success(f"NaNs en la columna '{selected_col}' tratados con el método: {metodo}")
                        st.session_state.df = df
                else:
                    st.success("🎉 No hay valores nulos en el dataset.")

        else:
            st.warning("No se ha subido ningún dataset aún.")
            st.info("Por favor, sube un dataset para proceder con el preprocesado.")

    if choice == "Detector Outliers":
        st.title("Detector de Outliers")
        if 'df' in locals():
            st.subheader("Deteccion de outliers con i-forest")
            column = st.selectbox("Selecciona una columna", st.session_state.df.columns)
            lags = st.number_input("Número de lags", min_value=1, max_value=100, value=24)
            contamination = st.number_input("Contaminación", min_value=0.0, max_value=1.0, value=0.05)
            st.markdown("**Nota:** El número de lags es el número de pasos hacia atrás que se utilizarán para crear las ventanas de datos. La contaminación es la proporción de puntos que se espera que sean outliers.")
            if st.button("Detectar Outliers con i-forest"):
                st.session_state.df.to_csv("dataset/df.csv")
                path="df.csv"
                with st.spinner("Detectando outliers..."):
                    result=subprocess.run([
                        "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                        "iForest.py",
                        str(column),
                        str(lags),
                        str(contamination),
                        str(st.session_state.time_col),
                        path
                    ],capture_output=True, text=True)
                    outliers = list(map(int, re.findall(r'\d+', result.stdout)))
                    st.session_state.outliers = outliers  # Save to session state
                    st.write("Outliers detectados:", outliers)
                img = Image.open('outliers_isolation_forest.png')
                st.image(img, caption='Outliers detectados')
                        
            st.subheader("Detección de outliers con un autonecoder")
            column_Autoencoder = st.selectbox("Selecciona una columna", st.session_state.df.columns,key="autoencoder_column")
            lags_Autoencoder = st.number_input("Número de lags", min_value=1, max_value=100, value=24,key="autoencoder_lags")
            percentile_Autoencoder = st.number_input("Percentile", min_value=0, max_value=100, step=1, value=95,key="autoencoder_percentile")
            if st.button("Detectar Outliers con Autoencoder"):
                st.session_state.df.to_csv("dataset/df.csv")
                path = "df.csv"
                with st.spinner("Detectando outliers..."):
                    result = subprocess.run([
                        "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",
                        "Autoencoder.py",
                        str(column_Autoencoder),
                        str(lags_Autoencoder),
                        str(percentile_Autoencoder),
                        str(st.session_state.time_col),
                        path
                    ], capture_output=True, text=True)
                    outliers = list(map(int, re.findall(r'\d+', result.stdout)))
                    st.session_state.outliers = outliers  # Save to session state
                    st.write("Outliers detectados:", outliers)
                img = Image.open('outliers_autoencoder.png')
                st.image(img, caption='Outliers detectados')

            st.subheader("Gestión de Outliers detectados")  
            print (st.session_state.outliers)
            st.markdown("1. Se puede interpretar que no existe ningun outlier, pues puede que los detectores sean sensibles a los datos y el ruido lo detecten como anomalia. \n")
            st.markdown("2. Puedes cambiar los outliers detectados por NaN para tratarlos en la sección de preprocesado previa.") 
            if st.button("Cambiar Outliers por NaN"):
                with st.spinner("Cambiando outliers por NaN..."):
                    valid_outliers = [i for i in st.session_state.outliers if i < len(st.session_state.df)]
                    st.session_state.df.iloc[valid_outliers, st.session_state.df.columns.get_loc(column_Autoencoder)] = np.nan
                    st.success("Outliers cambiados por NaN exitosamente.")
                    st.dataframe(st.session_state.df.head(n=70))
                    st.markdown("**Nota:** Los outliers han sido reemplazados por NaN. Puedes proceder a tratarlos en la sección de preprocesado.")
                    st.session_state.outliers = []  # Limpiar la lista de outliers
        else:
            st.warning("No se ha subido ningún dataset aún.")
            st.info("Por favor, sube un dataset para proceder con el preprocesado.")



    if choice=="Graficas Preprocesado":
        st.title("Graficas Preprocesado")
        if 'df' in locals() and len(st.session_state.nan_cols)==0:
            st.subheader("Fast Fourier Transform")
            fig,top3,tiempo,mult,tiempo2=fft(st.session_state.df[st.session_state.target_col], st.session_state.frecuencia)
            st.plotly_chart(fig)
            for i in range(3):
                st.text(f"Pico {i+1}: {top3[i]:.4f} ciclos/{tiempo} = {(top3[i] * mult):.4f} ciclos/{tiempo2}")

            st.markdown(texto_test())

            N_lags_ADF=st.selectbox("Seleccina el número de lags para ADFuller",["Manual", "Auto"])
            if N_lags_ADF=="Manual":
                lags_ADF=st.number_input("Núm lags", min_value=1, max_value=100,key="ADF_lags")
            else:
                lags_ADF=None
            
            regression_ADF=st.selectbox("Regresión constante(c) // Regresión constante con tendencia(ct)",["c","ct"])
            if st.button("Realizar test de Dickey-Fuller"):
                result = test_adfuller(st.session_state.df[st.session_state.target_col], lags_ADF,regression_ADF)
                st.session_state.adf_result = result

            if st.session_state.adf_result is not None:
                result = st.session_state.adf_result
                st.text(f"ADF Statistic: {result[0]}\n\nP-Value: {result[1]}")
                for key, value in result[4].items():
                    st.text('Critical Values:\t%s: %.3f' % (key, value))
                if (result[1] <= 0.05) & (result[0] < result[4]["1%"]):
                    st.success("Rechaza la hipótesis - La serie es estacionaria.")
                else:
                    st.warning("Fallo al rechazar la hipótesis - La serie no es estacionaria.")

            st.subheader("Test KPSS")

            N_lags_KPSS=st.selectbox("Seleccina el número de lags para KPSS",["Manual", "Auto"])
            if N_lags_KPSS=="Manual":
                lags_KPSS=st.number_input("Núm lags", min_value=1, max_value=100,key="kpss_lags")
            else:
                lags_KPSS="auto"

            regression_KPSS=st.selectbox("Regresión",["c","ct"])
            if st.button("Realizar test de Kwiatkowski-Phillips-Schmidt-Shin"):
                result = test_kpss(st.session_state.df[st.session_state.target_col], lags_KPSS,regression_KPSS)
                st.session_state.kpss_result = result

            if st.session_state.kpss_result is not None:
                result = st.session_state.kpss_result
                st.text(f"KPSS Statistic: {result[0]}\n\nP-Value: {result[1]}")
                for key, value in result[3].items():
                    st.text('Critical Values:\t%s: %.3f' % (key, value))
                if (result[1] <= 0.05) & (result[0] > result[3]["1%"]):
                    st.warning("Rechaza la hipótesis - La serie NO es estacionaria.")
                else:
                    st.success("Fallo al rechazar la hipótesis - La serie ES estacionaria.")

            st.subheader("Gráficas ACF y PACF")
            lags=st.number_input("Número de lags para ACF y PACF", min_value=1, max_value=100)
            if st.button("Graficar ACF y PACF"):
                st.session_state.imagen_acf_pacf,st.session_state.acf,st.session_state.pacf=plot_acf_pacf(st.session_state.df[st.session_state.target_col],lags)
                st.pyplot(st.session_state.imagen_acf_pacf, use_container_width=True)
            elif st.session_state.imagen_acf_pacf:
                st.pyplot(st.session_state.imagen_acf_pacf)
            st.subheader("Estacionalidad calculada con STL (Estacionalidad Flexible)")
            lags_STL=st.number_input("Número de lags para STL", min_value=1, max_value=100)
            if st.button("Realizar descomposición STL"):
                st.subheader("Descomposición STL")
                fig=seasonal_SLT(st.session_state.df, st.session_state.time_col, st.session_state.target_col, lags_STL)
                st.plotly_chart(fig)
                st.session_state.STL=fig
            elif st.session_state.STL:
                st.plotly_chart(st.session_state.STL)
            st.subheader("Estacionalidad calculada con Seasonal Decompose(Estacionalidad Fija)")
            lags_seasonal_descompose=st.number_input("Número de lags para Seasonal Decompose", min_value=1, max_value=100)
            if st.button("Realizar Seasonal Decompose"):
                st.subheader("Seasonal Decompose")
                st.plotly_chart(descompose_seasonal(st.session_state.df, st.session_state.target_col, lags_seasonal_descompose))
                st.session_state.imagen_Seasonal_Decompose=descompose_seasonal(st.session_state.df, st.session_state.target_col, lags_seasonal_descompose)
            elif st.session_state.imagen_Seasonal_Decompose:
                st.plotly_chart(st.session_state.imagen_Seasonal_Decompose)

            st.title("💬 Chat con documentos usando AnythingLLM")
            user_question = st.text_input("Escribe tu pregunta (Tambien se envian las graficas y resultados de los test hechos):")
            if user_question:
                with st.spinner("Consultando en N8N..."):
                    data = {
                        "resultados KPPS": st.session_state.kpss_result,
                        "resultados Dickey-Fuller": st.session_state.adf_result,
                        "resultados_fft": f"Pico {i+1}: {top3[i]:.4f} ciclos/{tiempo} = {(top3[i] * mult):.4f} ciclos/{tiempo2}",
                        "pacf":st.session_state.pacf,
                        "acf":st.session_state.acf,
                        "mensaje": user_question
                    }
                    response = requests.post("http://localhost:5678/webhook-test/python-a-n8n", data=data)

                    if response.status_code == 200:
                        data = response.text
                        st.markdown("### Respuesta:")
                        st.write(data)
                    else:
                        st.error(f"Error: {response.status_code}")


        else:
            st.warning("No se ha subido ningún dataset aún o hay columnas con NaNs.")
            st.info("Por favor, sube un dataset y realiza el preprocesado para proceder con las gráficas.")

    if choice == "Solucionar la NO ESTACIONARIEDAD":
        st.title("Tecnicas para solucionar la NO ESTACIONARIEDAD:")
        st.subheader("1. Diferenciación")
        st.markdown("La diferenciación es una técnica común para hacer que una serie temporal sea estacionaria. Consiste en restar el valor actual del valor anterior, eliminando así tendencias y estacionalidades. Si se detecta algun patron estacional (cada 24h/7 dias...) se puede aplicar diferenciación estacional poniendo el número de lags correspondiente.")
        diff=int(st.number_input("Seleccione usar diferenciación estacional. Se puede aplicar tantas veces como quiera el usuario", min_value=1, step=1, value=1, key="diff"))
        if st.button("Aplicar diferenciación"):
            if 'df' in locals():
                df = st.session_state.df.copy()
                datos_fill= df[st.session_state.target_col][:diff]
                df[st.session_state.target_col] = df[st.session_state.target_col].diff(periods=diff).fillna(datos_fill)
                st.success(f"Diferenciacion con t-{diff} aplicada exitosamente.")
                st.dataframe(df.head(n=70))
                st.session_state.df = df
        st.subheader("2. Matematica")
        if st.button("Aplicar transformación logarítmica. **Nota: Cuidado con datos negativos o 0**"):
            if 'df' in locals():
                df = st.session_state.df.copy()
                df = df[st.session_state.target_col].apply(lambda x: log(x) if x > 0 else 0)
                st.success("Transformación logarítmica aplicada exitosamente.")
                st.dataframe(df.head(n=70))
                st.session_state.df = df
            else:
                st.warning("No se ha subido ningún dataset aún.")
                st.info("Por favor, sube un dataset para proceder con la transformación.")
        if st.button("Aplicar transformación raíz cuadrada"):
            if 'df' in locals():
                df = st.session_state.df.copy()
                df[st.session_state.target_col] = df[st.session_state.target_col].apply(lambda x: sqrt(x) if x > 0 else 0)
                st.success("Transformación raíz cuadrada aplicada exitosamente.")
                st.dataframe(df.head(n=70))
                st.session_state.df = df
            else:
                st.warning("No se ha subido ningún dataset aún.")
                st.info("Por favor, sube un dataset para proceder con la transformación.")
        st.subheader("Una vez aplicada la diferenciación o transformación deseada, puedes volver a la sección 'Graficas Preprocesado' realizar los tests de estacionariedad (Dickey-Fuller y KPSS) para verificar si la serie es ahora estacionaria.")
    # Modelado automático con PyCaret
    if choice == "Modelado":
        
        st.title("Machine Learning Modelling")
        if 'df' in locals():
            st.header("🔹 Modelo LSTM")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_LSTM = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8)
                vl_size_LSTM = st.number_input("👍🏼 Porcentaje de Validacion", min_value=float(0),max_value=float(1), value=0.1)
                steps_LSTM = int(st.number_input("👣 Pasos para comparar predicciones con el valor real", step=1, value=168))
            with col2:
                output_length_LSTM = int(st.number_input("⏳ Pasos a predecir",step=1, value=2))
                epochs_LSTM = int(st.number_input("🔄 Numero de iteraciones", step=1, value=100))
                batch_size_LSTM = int(st.number_input("⏬ Tamaño de Batch", step=1, value=16))
                dropout_LSTM = float(st.number_input("💣 Porcentaje de Dropout", min_value=0.0, max_value=1.0, value=0.3, step=0.01))
                
            with col3:
                lags_LSTM = int(st.number_input("📊 Número de lags", step=1, value=24))
                layers_LSTM = int(st.number_input("🧬 Número de capas", step=1, value=3))
                units_LSTM = st.text_input("🧠 Escribe el número de neuronas de cada capa:", value="40")
                st.text("Ejemplo: 40, 30, 20 para 3 capas con 40, 30 y 20 neuronas respectivamente. (Ya se creará la última capa de salida con x neuronas= x Pasos a predecir)")


            st.markdown("---")  # Línea divisoria
            st.markdown("---")

            st.header("🔹 Modelo XGBOOST")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_XGB = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8, key='XGB1')
                vl_size_XGB = st.number_input("👍🏼 Porcentaje de Validacion", min_value=float(0),max_value=float(1), value=0.1, key='XGB2')
                steps_XGB = int(st.number_input("👣 Pasos para comparar predicciones con el valor real", step=1, value=168, key='XGB3'))
                depth_XGB = int(st.number_input("📏 Profundidad máxima de los árboles", step=1, value=5, key='XGB11'))
            with col2:
                output_length_XGB = int(st.number_input("⏳ Pasos a predecir",step=1, value=2, key='XGB4'))
                early_stopping_XGB = int(st.number_input("🛑 Iteraciones Early Stopping", step=1, value=5, key='XGB5'))
                tree_XGB = int(st.number_input("🌳 Número de árboles", step=5, value=100, key='XGB6'))
                learning_rate_XGB = float(st.number_input("📖 Tasa de aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key='XGB7'))

            with col3:
                lags_XGB = int(st.number_input("📊 Número de lags", step=1, value=24, key='XGB8'))
                col_samples_XGB = st.number_input("➡️ Porcentaje de FILAS usadas para entrenar cada árbol", min_value=float(0),max_value=float(1), value=0.8, key='XGB9')
                sub_sample_XGB = st.number_input("⬆️ Porcentaje de COLUMNAS usadas para entrenar cada árbol", min_value=float(0),max_value=float(1), value=0.8, key='XGB10')

            st.markdown("---") 
            st.markdown("---")

            st.header("🔹 Modelo CNN-LSTM-AM")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_CNN_LSTM_AM = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8,key='CNN_LSTM_AM1')
                vl_size_CNN_LSTM_AM = st.number_input("👍🏼 Porcentaje de Validacion", min_value=float(0),max_value=float(1), value=0.1,key='CNN_LSTM_AM2')
                steps_CNN_LSTM_AM = int(st.number_input("👣 Pasos para comparar predicciones con el valor real", step=1, value=168,key='CNN_LSTM_AM3'))
                filters_CNN_LSTM_AM = st.text_input("🔎 Número de filtros", value=120, key='CNN_LSTM_AM8')
                st.text("Ejemplo: 40, 30, 20 para 3 capas con 40, 30 y 20 neuronas respectivamente.")
            with col2:
                output_length_CNN_LSTM_AM = int(st.number_input("⏳ Pasos a predecir",step=1, value=2, key='CNN_LSTM_AM4'))
                epochs_CNN_LSTM_AM = int(st.number_input("🔄 Numero de iteraciones", step=1, value=100, key='CNN_LSTM_AM5'))
                batch_size_CNN_LSTM_AM = int(st.number_input("⏬ Tamaño de Batch", step=1, value=16, key='CNN_LSTM_AM6'))
                dropout_CNN_LSTM_AM = float(st.number_input("💣 Porcentaje de Dropout", min_value=0.0, max_value=1.0, value=0.3, step=0.01, key='CNN_LSTM_AM7'))
            with col3:
                lags_CNN_LSTM_AM = int(st.number_input("📊 Número de lags", step=1, value=24, key='CNN_LSTM_AM12'))
                layers_Conv_CNN_LSTM_AM = int(st.number_input("🧅 Número de capas CNN", step=1, value=3, key='CNN_LSTM_AM9'))
                layers_Long_short_CNN_LSTM_AM = int(st.number_input("🧬 Número de capas LSTM", step=1, value=3, key='CNN_LSTM_AM10'))
                units_CNN_LSTM_AM = st.text_input("🧠Número de neuronas", value=40, key='CNN_LSTM_AM11')
                st.text("Ejemplo: 40, 30, 20 para 3 capas con 40, 30 y 20 neuronas respectivamente. (Ya se creará la última capa de salida con x neuronas= x Pasos a predecir)")

            st.markdown("---")  # Línea divisoria
            st.markdown("---")

            st.header("🔹 Modelo MLP")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_MLP = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8,key='MLP1')
                vl_size_MLP = st.number_input("👍🏼 Porcentaje de Validacion", min_value=float(0),max_value=float(1), value=0.1,key='MLP2')
                steps_MLP = int(st.number_input("👣 Pasos para comparar predicciones con el valor real", step=1, value=168,key='MLP3'))

            with col2:
                output_length_MLP = int(st.number_input("⏳ Pasos a predecir",step=1, value=2, key='MLP4'))
                epochs_MLP = int(st.number_input("🔄 Numero de iteraciones", step=1, value=100, key='MLP5'))
                batch_size_MLP = int(st.number_input("⏬ Tamaño de Batch", step=1, value=16, key='MLP6'))
                dropout_MLP = float(st.number_input("💣 Porcentaje de Dropout", min_value=0.0, max_value=1.0, value=0.3, step=0.01, key='MLP7'))
            with col3:
                lags_MLP = int(st.number_input("📊 Número de lags", step=1, value=24, key='MLP12'))
                layers_Dense_MLP = int(st.number_input("🧅 Número de capas densas", step=1, value=3, key='MLP9'))
                units_MLP = st.text_input("🧠Número de neuronas", value=40, key='MLP11')
                st.text("Ejemplo: 40, 30, 20 para 3 capas con 40, 30 y 20 neuronas respectivamente. (Ya se creará la última capa de salida con x neuronas= x Pasos a predecir)")

            st.markdown("---")  # Línea divisoria
            st.markdown("---")

            st.header("🔹 Modelo SARIMAX")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_SARIMAX = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8,key='SARIMAX1')

            with col2:
                seasonality_SARIMAX = bool(st.selectbox("📅 Estacionalidad", ["True", "False"], key='SARIMAX2'))  
                steps_SARIMAX = int(st.number_input("👣 Pasos para comparar predicciones con el valor real", step=1, value=168,key='SARIMAX3'))
            with col3:
                lags_SARIMAX = int(st.number_input("📊 Número de lags", step=1, value=24, key='SARIMAX12'))


            st.markdown("---")  # Línea divisoria
            st.markdown("---")

            st.header("🔹 Modelo NEURAL PROPHET")
            col1, col2, col3 = st.columns(3)
            with col1:
                tr_size_NP = st.number_input("🏋️ Porcentaje de entrenamiento", min_value=float(0),max_value=float(1), value=0.8,key='NP1')
                vl_size_NP = st.number_input("👍🏼 Porcentaje de Validacion", min_value=float(0),max_value=float(1), value=0.1,key='NP2')
                steps_NP = int(st.number_input("👣 Pasos para comparar predicciones con el valor real*Cuidado, este modelo tarda mucho*", step=1, value=168,key='NP3'))
                learning_rate_NP = float(st.number_input("📖 Tasa de aprendizaje", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key='NP4'))

            with col2:
                output_length_NP = int(st.number_input("⏳ Pasos a predecir",step=1, value=2, key='NP7'))
                epochs_NP = int(st.number_input("🔄 Numero de iteraciones", step=1, value=100, key='NP5'))
                lags_NP = int(st.number_input("📊 Número de lags", step=1, value=24, key='NP12'))

            with col3:
                yearly_seasonality_NP = bool(st.selectbox("📅 Estacionalidad ANUAL", ["True", "False"], key='NP8'))
                weekly_seasonality_NP = bool(st.selectbox("⏳ Estacionalidad SEMANAL", ["True", "False"], key='NP9'))
                daily_seasonality_NP = bool(st.selectbox("🕐 Estacionalidad DIARIA", ["True", "False"], key='NP10'))



            st.markdown("---")
            st.markdown("---")  # Línea divisoria
            # Cuadro de selección de modelos
            st.markdown("### 🎛️Selecciona los modelos que quieres ejecutar:")
            LSTM = st.checkbox("LSTM")
            XGBOOST = st.checkbox("XGBOOST")
            CNN_LSTM_AM = st.checkbox("CNN_LSTM_AM")
            MLP = st.checkbox("MLP")
            SARIMAX = st.checkbox("SARIMAX")
            NEURALPROPHET = st.checkbox("NEURALPROPHET")
            # Botón para ejecutar
            if st.button("Ejecutar modelos seleccionados"):
                total = sum([LSTM, XGBOOST, CNN_LSTM_AM, MLP, SARIMAX, NEURALPROPHET])
                progreso = st.progress(0)
                ejecutados = 0
                st.session_state.df.to_csv("dataset/df.csv")
                path="df.csv"
                if LSTM:
                    with st.spinner("Ejecutando LSTM..."):
                        subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "LSTM.py",
                            str(tr_size_LSTM),
                            str(vl_size_LSTM),
                            str(st.session_state.target_col),
                            str(lags_LSTM),
                            str(output_length_LSTM),
                            str(epochs_LSTM),
                            str(batch_size_LSTM),
                            str(layers_LSTM),
                            str(units_LSTM),
                            str(dropout_LSTM),
                            str(steps_LSTM),
                            str (st.session_state.time_col),
                            path
                        ])
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    LSTM_name="LSTM_RESULTS_"+fecha
                    shutil.make_archive(LSTM_name, 'zip', "C:/Users/Noel/Desktop/TFM/LSTM")
                    st.session_state["rutas_zip"]["LSTM"] = f'{LSTM_name}.zip'
                    progreso.progress(ejecutados / total)

                if XGBOOST:
                    with st.spinner("Ejecutando XGBOOST..."):
                        subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "XGB.py",
                            str(tr_size_XGB),
                            str(vl_size_XGB),
                            str(st.session_state.target_col),
                            str(lags_XGB),
                            str(output_length_XGB),
                            str(early_stopping_XGB),
                            str(tree_XGB),
                            str(depth_XGB),
                            str(learning_rate_XGB),
                            str(sub_sample_XGB),
                            str(col_samples_XGB),
                            str(steps_XGB),
                            str (st.session_state.time_col),
                            path
                        ])
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    XGB_name="XGB_RESULTS_"+fecha
                    shutil.make_archive(XGB_name, 'zip', "C:/Users/Noel/Desktop/TFM/XGBOOST")
                    st.session_state["rutas_zip"]["XGBOOST"] = f'{XGB_name}.zip'
                    progreso.progress(ejecutados / total)

                if CNN_LSTM_AM:
                    with st.spinner("Ejecutando CNN_LSTM_AM..."):
                        subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "CNN_LSTM_AM.py",
                            str(tr_size_CNN_LSTM_AM),
                            str(vl_size_CNN_LSTM_AM),
                            str(st.session_state.target_col),
                            str(lags_CNN_LSTM_AM),
                            str(output_length_CNN_LSTM_AM),
                            str(layers_Conv_CNN_LSTM_AM),
                            str(layers_Long_short_CNN_LSTM_AM),
                            str(units_CNN_LSTM_AM),
                            str(filters_CNN_LSTM_AM),
                            str(dropout_CNN_LSTM_AM),
                            str(batch_size_CNN_LSTM_AM),
                            str(epochs_CNN_LSTM_AM),
                            str(steps_CNN_LSTM_AM),
                            str (st.session_state.time_col),
                            path
                        ])
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    CNN_LSTM_AM_name="CNN_LSTM_AM_RESULTS_"+fecha
                    shutil.make_archive(CNN_LSTM_AM_name, 'zip', "C:/Users/Noel/Desktop/TFM/CNN_LSTM_AM")
                    st.session_state["rutas_zip"]["CNN_LSTM_AM"] = f'{CNN_LSTM_AM_name}.zip'
                    progreso.progress(ejecutados / total)

                if MLP:
                    with st.spinner("Ejecutando MLP..."):
                        subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "MLP.py",
                            str(tr_size_MLP),
                            str(vl_size_MLP),
                            str(st.session_state.target_col),
                            str(lags_MLP),
                            str(output_length_MLP),
                            str(epochs_MLP),
                            str(batch_size_MLP),
                            str(layers_Dense_MLP),
                            str(units_MLP),
                            str(dropout_MLP),
                            str(steps_MLP),
                            str (st.session_state.time_col),
                            path
                        ])
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    MLP_name="MLP_RESULTS_"+fecha
                    shutil.make_archive(MLP_name, 'zip', "C:/Users/Noel/Desktop/TFM/MLP")
                    st.session_state["rutas_zip"]["MLP"] = f'{MLP_name}.zip'
                    progreso.progress(ejecutados / total)

                if SARIMAX:
                    with st.spinner("Ejecutando SARIMAX..."):
                        crear_SARIMAX(st.session_state.df, tr_size_SARIMAX, lags_SARIMAX, st.session_state.target_col, seasonality_SARIMAX, steps_SARIMAX)
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    SARIMAX_name="SARIMAX_RESULTS_"+fecha
                    shutil.make_archive(SARIMAX_name, 'zip', "C:/Users/Noel/Desktop/TFM/SARIMAX")
                    st.session_state["rutas_zip"]["SARIMAX"] = f'{SARIMAX_name}.zip'
                    progreso.progress(ejecutados / total)

                if NEURALPROPHET:
                    with st.spinner("Ejecutando NEURALPROPHET..."):
                        subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "NEURALPROPHET.py",
                            str(tr_size_NP),
                            str(vl_size_NP),
                            str(st.session_state.target_col),
                            str(lags_NP),
                            str(output_length_NP),
                            str(epochs_NP),
                            str(st.session_state.time_col),
                            str(st.session_state.format_date),
                            str(st.session_state.frecuencia),
                            str(learning_rate_NP),
                            str(yearly_seasonality_NP),
                            str(weekly_seasonality_NP),
                            str(daily_seasonality_NP),
                            str(steps_NP),
                            path
                        ])
                    ejecutados += 1
                    now = datetime.now()
                    fecha = now.strftime("%Y%m%d_%H%M%S")
                    NEURALPROPHET_name="NEURALPROPHET_RESULTS_"+fecha
                    shutil.make_archive(NEURALPROPHET_name, 'zip', "C:/Users/Noel/Desktop/TFM/NEURALPROPHET")
                    st.session_state["rutas_zip"]["NEURALPROPHET"] = f'{NEURALPROPHET_name}.zip'
                    progreso.progress(ejecutados / total)

            else:
                st.warning("No se ha subido ningún dataset aún.")
            st.info("Por favor, sube un dataset para proceder con el modelado.")

    # Descargar modelo
    if choice == "Descargar modelo":
        if "rutas_zip" in st.session_state and st.session_state["rutas_zip"]:
            claves_a_borrar = []

            for modelo, ruta in st.session_state["rutas_zip"].items():
                with open(ruta, "rb") as f:
                    data = f.read()

                if st.download_button(
                    label=f"📦 Descargar {modelo}",
                    data=data,
                    file_name=ruta,
                    mime="application/zip",
                    key=f"descargar_{modelo}"
                ):
                    claves_a_borrar.append(modelo)

            # Borrar solo los que se descargaron
            for modelo in claves_a_borrar:
                del st.session_state["rutas_zip"][modelo]
        else:
            st.warning("No se ha creado ningún modelo aún.")
            st.info("Por favor, sube un dataset y ejecuta el modelado para generar un modelo antes de descargarlo.")

if selected == "Clustering":
    
    with st.sidebar:
        st.image("https://www.udc.es/export/sites/udc/epef/.galleries/EPEF/LogoEPEF.jpg_2063069294.jpg")
        st.title("Clustering de Series Temporales")
        choice2 = st.radio("Selección", ["Subir datos", "Eliminar Nans", "Grafica Elbow con TimeSeriesKMeans", "Seleccion K óptimo"])
        st.info("Aplicación para clustering de series temporales.")
        st.info("Desarrollado por:")
        st.info("Noel Freire Mahía")
    
    if choice2 == "Subir datos":
        st.title("Subir datos para Clustering")
        st.info("Sube un archivo CSV con las series temporales que deseas agrupar.")
        # Listar archivos disponibles en la carpeta data
        files = [f for f in os.listdir("dataset") if f.endswith(('.csv', '.xlsx', '.json'))]

        if files:
            file_selected = st.selectbox("Selecciona un archivo en la carpeta 'dataset/'", files)
            
            if st.button("Cargar archivo"):
                st.session_state.clear()
                file_path = os.path.join("dataset", file_selected)
                file_type = file_selected.split('.')[-1]

                with st.spinner("Cargando el archivo..."):
                    if file_type == "csv":
                        df = pd.read_csv(file_path)
                    elif file_type in ["xlsx", "xls"]:
                        df = pd.read_excel(file_path)
                    elif file_type == "json":
                        df = pd.read_json(file_path)
                    else:
                        st.error("Tipo de archivo no soportado.")
                        df = None

                if 'df' in locals():
                    st.success("Archivo cargado exitosamente")
                    st.dataframe(df)
                    # Guardar si quieres reutilizar
                    df.to_csv("dataset.csv", index=False)
                    st.session_state.df=df
            else:
                st.warning("No hay archivos en la carpeta 'data/'.")
                st.info("Por favor, coloca un archivo grande en la carpeta 'data/' y actualiza la página.")
    if choice2== "Eliminar Nans":
        st.title("Eliminar valores nulos")
        df = st.session_state.df
        st.dataframe(st.session_state.df.head(n=70))
        nan_cols = df.columns[df.isna().any()].tolist()# Verificar si hay valores nulos
        st.session_state.nan_cols = nan_cols
        if nan_cols:
            st.markdown("Estas columnas tienen valores nulos:")
            st.write(df[nan_cols].isna().sum())
            selected_col = st.selectbox("Selecciona una columna para tratar NaNs", nan_cols)
            metodo = st.selectbox("Selecciona el método para tratar los NaNs", [
                "Forward Fill (ffill)",
                "Backward Fill (bfill)",
                "Interpolación",
                "Rellenar con 0",
                "Rellenar con la media",
                "Rellenar con la mediana",
                "Eliminar filas con NaNs"
            ])
            if st.button("Aplicar tratamiento"):
                if metodo == "Forward Fill (ffill)":
                    df[selected_col] = df[selected_col].fillna(method='ffill')
                elif metodo == "Backward Fill (bfill)":
                    df[selected_col] = df[selected_col].fillna(method='bfill')
                elif metodo == "Interpolación":
                    df[selected_col] = df[selected_col].interpolate()
                elif metodo == "Rellenar con 0":
                    df[selected_col] = df[selected_col].fillna(0)
                elif metodo == "Rellenar con la media":
                    media = df[selected_col].mean()
                    df[selected_col] = df[selected_col].fillna(media)
                elif metodo == "Rellenar con la mediana":
                    mediana = df[selected_col].median()
                    df[selected_col] = df[selected_col].fillna(mediana)
                elif metodo == "Eliminar filas con NaNs":
                    df = df.dropna(subset=[selected_col])
                
                st.success(f"NaNs en la columna '{selected_col}' tratados con el método: {metodo}")
                st.session_state.df = df
        else:
            st.success("🎉 No hay valores nulos en el dataset.")

    if choice2 == "Grafica Elbow con TimeSeriesKMeans":
        st.title("Gráfica Elbow con TimeSeriesKMeans")
        if 'df' in locals():
            st.subheader("Selecciona la columna de series temporales")
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("Columna", df.columns)
                lags = int(st.number_input("Como se agrupan los datos", step=1, value=24))
                clusters= int(st.number_input("Número de clusters a probar", step=1, value=10))
                amount = int(st.number_input("Cantidad de grupos a utilizar", step=1, value=100))
            if st.button("Generar gráfica Elbow"):
                st.session_state.df.to_csv("dataset/df.csv")
                path="df.csv"
                st.session_state.col_ref=selected_col
                st.session_state.lags=lags
                st.session_state.cant=amount
                with st.spinner("Generando gráfica..."):
                    subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "Elbow.py",
                            str(selected_col),
                            str(lags),
                            str(clusters),
                            str(amount),
                            path
                    ])
                    st.success("Gráfica generada exitosamente.")
                    img = Image.open('elbow_plot.png')
                    st.image(img, caption='Gráfica Elbow')
                    st.markdown("El usuario tiene que seleccionar con que valor punto donde la pendiente comienza a disminuir significativamente.")
                    st.markdown("Ese valor de K (Eje X) se debera introducir en el siguiente apartado")
                    st.markdown(st.session_state.cant)
        else:
            st.warning("No se ha subido ningún dataset aún.")
            st.info("Por favor, sube un dataset para proceder con la gráfica Elbow.")

    if choice2== "Seleccion K óptimo":
        st.title("Gráfica TimeSeriesKMeans de los diferentes grupos")
        if 'df' in locals():
            st.subheader("Selecciona la columna de series temporales")
            col1, col2 = st.columns(2)
            with col1:
                clusters= int(st.number_input("Número de clusters óptimo", step=1, value=10, key='clusters'))
            if st.button("Generar gráfica Clusters"):
                st.session_state.df.to_csv("dataset/df.csv")
                path="df.csv"
                with st.spinner("Generando gráfica..."):
                    subprocess.run([
                            "C:/Users/Noel/Desktop/TFM/TensorFlow_env/Scripts/python.exe",  
                            "Cluster.py",
                            str(st.session_state.col_ref),
                            str(st.session_state.lags),
                            str(clusters),
                            str(st.session_state.cant),
                            path
                    ])
                    st.success("Gráfica generada exitosamente.")
                    img = Image.open('centroides.png')
                    st.image(img, caption='Gráfica Centroides')


