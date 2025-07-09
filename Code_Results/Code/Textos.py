def texto_intro1():
    texto="""
        <h1>¡Hola! 👋 Esta es la página de tratamiento de series temporales</h1>
        <p>Da aquí tus primeros pasos en el ámbito de la inteligencia artificial, concretamente en el campo de las series temporales.
        Esta página también es ideal para expertos en estos campos, pero si este no es tu caso, existe un modelo RAG que será tu Sherlock 😉.</p>

        <h3>¿Qué es exactamente esto❓</h3>
        <p>Este entorno es una aplicación web diseñada para ayudar al usuario a entrenar modelos sin necesidad de escribir una línea
        de código. Es bastante intuitiva, de manera que crear un modelo desde cero es sencillo tanto para expertos como para principiantes. ¡Empecemos!</p>

        <h3>¿Qué se puede hacer❓</h3>
        <p>En esta página se simplifican las etapas de preprocesamiento y entrenamiento de modelos de predicción de series temporales. De esta manera,
        el usuario puede ahorrar tiempo y esfuerzo evitando aspectos técnicos de programación, permitiéndole centrarse en interpretar los resultados
        y tomar decisiones. Esta web está pensada para que cualquier usuario, con o sin experiencia en el ámbito de la programación o la inteligencia artificial, pueda utilizarla.</p>
        """
    return texto

def texto_intro2():
    texto="""
            <h1>¿Qué se puede hacer en esta página?</h1>
            <p>En esta página puedes entrenar un modelo en unos pocos pasos, que son:</p>

            <h3>Subir el dataset deseado 📑:</h3>
            <p>Primero debes añadir el dataset a la carpeta "datasets", ubicada en el mismo path que este archivo. Son válidos los archivos <b>CSV</b>, <b>XLSX</b>, <b>XLS</b> y <b>JSON</b>.</p>

            <h3>Informe detallado de las covariables 📋:</h3>
            <p>Una vez subido el dataset, podrás consultar sus características en la sección "Información general de todas las covariables".
            Toda esta sección utiliza <b>ProfileReport</b> del paquete de la librería <b>ydata_profiling</b>.</p>

            <h3>Detector de outliers ⚠️:</h3>
            <p>Permite buscar valores anómalos en la columna seleccionada del dataset. Existen dos posibles detectores: autoencoder e iforest.</p>

            <h3>Preprocesado de las series temporales ⚙️:</h3>
            <p>Se ajusta el dataset, haciendo especial hincapié en la columna de fecha para obtener gráficas que ayuden en la correcta selección del modelo e hiperparámetros.</p>

            <h3>Gráficas del preprocesado 📊:</h3>
            <p>En esta sección se pueden analizar diversas gráficas (descomposición STL, FFT, ACF...) para seleccionar la mejor técnica de predicción o consultar al modelo RAG.</p>

            <h3>Problema de estacionariedad ❌:</h3>
            <p>Se abordará el problema de la estacionariedad (soluciones: diferenciación o transformaciones comunes), ya que si el dataset no es estacionario, muchos modelos no pueden realizar predicciones precisas.</p>

            <h3>Entrenamiento del modelo 🧮:</h3>
            <p>En la sección de modelado, podrás elegir los hiperparámetros de cada modelo y proceder a su entrenamiento.
            El sistema entrenará, comparará distintos modelos y seleccionará el que mejor rendimiento tenga.</p>

            <h3>Descarga de los modelos 💾:</h3>
            <p>Después de todo este proceso, podrás descargar tantos archivos <b>ZIP</b> como modelos entrenados. Estos incluirán los modelos, gráficas de error y otras imágenes que muestran el rendimiento de cada modelo. La elección del modelo óptimo quedará a criterio del usuario, ya que existen diversos criterios de selección (tiempo de entrenamiento, menor error, tamaño del modelo, etc.).</p>

            <p>A continuación, encontrarás un pequeño artículo que explica términos complejos como series temporales y forecasting de manera sencilla y amena:</p>
            """
    return texto

def texto_test():
    texto="""
    # Comprobación de Estacionalidad con la Prueba de Dickey-Fuller

    - **ADF > Valor crítico** ==> **No estacionario**
    - **ADF < Valor crítico** ==> **Estacionario**

    ## Otra comprobación:

    - **P-Value > 0.05** ==> la serie es **NO estacionaria**
    - **P-Value < 0.05** ==> la serie es **estacionaria**

    En teoría, siempre deberían coincidir, pero si no es el caso, se pasaría a usar otras pruebas como el **PP-Test** o el **KPSS**.
    """
    return texto

def texto_frecuencia():
    texto="""
            **¿Qué frecuencia tiene el dataset?**  
            Escribir la abreviatura que aparece a la derecha:

            - Segundo: S  
            - Minuto: T 
            - 10 Minutos: 10min 
            - Hora: H  
            - Día: D  
            - Mes: M  
            - Año: Y  
            """
    return texto
