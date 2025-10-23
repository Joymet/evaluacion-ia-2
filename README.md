# evaluacion-ia-2
Evaluación-2-Con-ReadME

**1) ¿Cuál es el umbral ideal para el modelo de predicción de diabetes?**

El umbral ideal para el modelo de predicción de diabetes se calcula utilizando métricas de desempeño, como la sensibilidad y la especificidad. 
Para este modelo, se utilizó el criterio de Youden (J), que maximiza la diferencia entre la tasa de verdaderos positivos (TPR) y la tasa de falsos positivos (FPR).
El umbral óptimo encontrado fue 0.5. Este umbral ofrece un equilibrio adecuado entre los errores de predicción, lo que es especialmente importante en escenarios de tamizaje de diabetes.


**2) ¿Cuáles son los factores que más influyen en el precio de los costos asociados al seguro médico?**

En el modelo de predicción de los costos del seguro médico, los factores más influyentes son:
Edad: A medida que las personas envejecen, los costos del seguro médico tienden a aumentar debido a una mayor probabilidad de enfermedades crónicas.
IMC (Índice de Masa Corporal): Un mayor IMC está relacionado con un mayor riesgo de enfermedades como la hipertensión y diabetes, lo que incrementa los costos de atención médica.
Tabaquismo: Fumar aumenta significativamente los costos debido a los riesgos asociados con enfermedades respiratorias y cardiovasculares.
Estos factores tienen una fuerte relación con los costos del seguro, siendo el IMC uno de los más relevantes.


**3) Hacer un análisis comparativo de cada característica de ambos modelos utilizando RandomForest.**

Al comparar Regresión Logística y RandomForest, observamos que ambos modelos muestran similitudes en algunas características clave, pero también diferencias:
Regresión Logística: Permite una interpretación clara de los coeficientes, donde las variables más influyentes son Glucosa y IMC. La relación es lineal y fácil de entender.
RandomForest: Aunque RandomForest no es tan interpretable, generalmente supera en precisión al modelo de regresión logística, especialmente en recall. Además, RandomForest maneja mejor las interacciones no lineales entre las características y es menos sensible a la sobreajuste.
En resumen, RandomForest tiende a tener una mayor precisión y recall, pero la Regresión Logística ofrece mayor interpretabilidad.


**4) ¿Qué técnica de optimización mejora el rendimiento de ambos modelos?**

Para mejorar el rendimiento de ambos modelos, se aplicaron dos técnicas de optimización:
Ajuste de Hiperparámetros: Se utilizó GridSearchCV para optimizar los parámetros del modelo, especialmente el valor de C (penalización) y el solver para la Regresión Logística. Esto ayudó a obtener mejores resultados de precisión.
Normalización de Datos: La normalización (con StandardScaler) de las características permitió que los modelos convergieran más rápidamente y mejoraron su capacidad de generalización.
Estas dos técnicas optimizan tanto la precisión como el desempeño general del modelo, especialmente para Regresión Logística.


**5) Explicar contexto de los datos.**

El conjunto de datos contiene información sobre pacientes y sus características médicas, con el objetivo de predecir si tienen diabetes o no.
El conjunto de datos incluye las siguientes características:

-Edad
-Glucosa
-Presión Arterial
-IMC (Índice de Masa Corporal)
-Historia Familiar de Diabetes
-Número de Embarazos (en mujeres)

La variable objetivo, outcome, es binaria (0 para no diabético y 1 para diabético), y es el objetivo principal de la predicción.
Este es un problema de clasificación binaria donde se busca identificar a los pacientes con mayor riesgo de desarrollar diabetes.


**6) Analizar el sesgo que presentan los modelos y explicar por qué.**

Los modelos de Regresión Logística y RandomForest presentan ciertos sesgos, especialmente hacia la clase positiva (diabéticos). 
Esto se debe a un desbalance en los datos, ya que la clase "no diabéticos" tiene menos ejemplos que la clase "diabéticos". 
El modelo tiene más facilidad para predecir correctamente a los pacientes diabéticos, pero tiene dificultades para identificar correctamente a los pacientes no diabéticos.

El sesgo se observa principalmente en la matriz de confusión, donde la clase "no diabéticos" muestra una menor tasa de recall y precisión.
Este desbalance puede ser mitigado utilizando técnicas como el rebalanceo de clases o ajustando el umbral de clasificación para mejorar el rendimiento en la clase minoritaria.






Parte 2. Funcionamieno del Algoritmo

🧠 ML Insurance and Diabetes Prediction

Este proyecto implementa dos modelos de *Machine Learning*:

1. Predicción de costos de seguro médico (modelo de regresión lineal)
2. Predicción de diabetes (modelo de clasificación con Regresión Logística)

Ambos modelos se integran en una aplicación web que permite realizar predicciones mediante una API desarrollada con FastAPI y una interfaz gráfica creada con Streamlit.

------------------------------------------------------------
🎯 Objetivos del proyecto

- Entrenar y evaluar dos modelos predictivos: uno para costos médicos y otro para diagnóstico de diabetes.
- Comparar el rendimiento y las variables más influyentes en cada modelo.
- Implementar técnicas de optimización y selección de umbral.
- Desplegar los modelos en un servicio web con interfaz visual interactiva.

------------------------------------------------------------
📂 Estructura del proyecto

ml-insurance-diabetes/
│
├── data/                      # Datos crudos y limpios
├── notebooks/                 # Jupyter Notebooks (EDA y entrenamiento)
├── src/                       # Scripts de entrenamiento y análisis
│   ├── train_diabetes.py
│   ├── train_regression.py
│   ├── rf_compare.py
│   └── optimize.py
│
├── app/                       # Backend (API) e interfaz gráfica (UI)
│   ├── api.py                 # FastAPI
│   └── ui.py                  # Streamlit
│
├── models/                    # Modelos entrenados (.pkl)
├── requirements.txt           # Dependencias
├── Dockerfile                 # Para ejecución en contenedor
└── README.md                  # Este documento

------------------------------------------------------------
⚙️ Requisitos

Antes de ejecutar la aplicación, asegúrate de tener instalado:

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, si clonas desde GitHub)
- Editor (Visual Studio Code recomendado)

Instala las dependencias ejecutando:

pip install -r requirements.txt

------------------------------------------------------------
🚀 Ejecución de la aplicación

1. Entrenar los modelos
   - Coloca tus datasets en la carpeta data/
     (por ejemplo diabetes.csv y insurance.csv).

   - Ejecuta los siguientes comandos:

     python src/train_diabetes.py
     python src/train_regression.py

   Esto generará los modelos entrenados dentro de la carpeta models/.

2. Ejecutar el Backend (API con FastAPI)

   cd app
   uvicorn api:app --reload

   El servidor se iniciará en:
   http://127.0.0.1:8000

   Puedes acceder a:
   - /docs → Documentación interactiva (Swagger)
   - /predict/diabetes → Endpoint para predecir diabetes
   - /predict/insurance → Endpoint para predecir costos médicos

3. Ejecutar la Interfaz Gráfica (Streamlit)

   streamlit run app/ui.py

   Se abrirá una ventana del navegador con la interfaz donde puedes ingresar valores como glucosa, edad, IMC, fumador, etc., y obtener las predicciones visualmente.

4. (Opcional) Ejecutar con Docker

   docker build -t ml-app .
   docker run -p 8000:8000 ml-app

------------------------------------------------------------
🧩 Detalles Técnicos de los Modelos

Modelo de Diabetes
- Algoritmo: Regresión Logística
- Umbral óptimo: 0.5 (criterio de Youden)
- Precisión general: 83%
- Características más influyentes: Glucosa, IMC, Edad y Función de Pedigrí

Modelo de Costos Médicos
- Algoritmo: Regresión Lineal / ElasticNet
- Factores más influyentes: Edad, IMC y Tabaquismo
- Métrica principal: MAE y RMSE

------------------------------------------------------------
🧠 Técnicas de Optimización Implementadas

- Normalización de características con StandardScaler
- Búsqueda de hiperparámetros con GridSearchCV
- Comparación de importancia de variables con RandomForest
- Evaluación cruzada (Cross-Validation) con KFold y RepeatedStratifiedKFold

------------------------------------------------------------
📊 Sesgo y Análisis de los Modelos

- Se detectó sesgo hacia la clase positiva (diabéticos), debido al desbalance en los datos.  
  El modelo tiende a clasificar más fácilmente a pacientes diabéticos que a no diabéticos.  
- Se recomienda ajustar el umbral o aplicar técnicas de rebalanceo de clases.  
- En el modelo de costos médicos, se observa un sesgo leve asociado al tabaquismo y la edad, ya que ambos factores tienden a incrementar el costo estimado.

------------------------------------------------------------
✅ Flujo Rápido de Ejecución

Paso | Comando | Descripción
------|----------|-------------
1 | pip install -r requirements.txt | Instala dependencias
2 | python src/train_diabetes.py | Entrena el modelo de diabetes
3 | python src/train_regression.py | Entrena el modelo de costos médicos
4 | uvicorn app.api:app --reload | Inicia la API backend
5 | streamlit run app/ui.py | Abre la interfaz web

------------------------------------------------------------
📘 Créditos

Desarrollado por Max (INACAP Valdivia)
Proyecto académico de integración para el módulo de Machine Learning y Despliegue Web.
Incluye análisis de modelos, comparación con RandomForest y optimización con GridSearchCV 
BASADOS EN LOS MODELOS DE KRAGGLE.
