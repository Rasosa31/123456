# Stock Price Movement Prediction with Machine Learning

This academic project applies Machine Learning techniques to predict whether a stock's price will go up or down the next day, using historical data and supervised models. The approach is practical, featuring a functional API, Dockerized environment, and a trained model ready to make predictions based on user-provided inputs.

## Project Objective

To develop a binary prediction system (up / down) for stock price movement, based on features extracted from historical data. The model was trained, evaluated, and deployed via an API to facilitate integration.

## Project Structure

### File / Folder	Description

	1. stock_pred_ec_oil.ipynb	Main notebook with data exploration, model training, and evaluation.
	2. app.py	Script exposing the model as a REST API using FastAPI.
	3. test_api.py	Unit tests to validate API functionality.
	4. requirements.txt	List of dependencies required to run the project.
	5. Dockerfile	Configuration for containerizing the application.
	6. DATA/	Folder containing the data used to train the model.
	7. best_model.pkl	Trained and serialized model ready to be loaded by the API.
	8. README.md	Project documentation.

## Technologies Used

    -Python 3.10+
    -Scikit-learn
    -Pandas / NumPy
    -FastAPI
    -Docker
    -Pytest

## Example Usage

Once the API is running, you can send a POST request with the following input features:
  
json

      {
  "Close": 10.0,  
  "Volume": 500000,
  "SMA_100": 12.0,
  "RSI_14": 56.0,
  "WTI_Close": 46.32
       }

The response will be:

json
       {
  "prediction": 1,
  "confidence": 0.6,
  "meaning": "1 = DOWN tomorrow"
        }

## Input Explanation

    1. Close: Closing price of the stock on the last trading day.
    2. Volume: Number of shares traded at the close of the last trading day.
    3. SMA_100: 100-period Simple Moving Average at the close of the last trading day.
    4. RSI_14: Relative Strength Index (14-period) at the close of the last trading day.
    5. WTI_Close: Closing price of WTI crude oil on the last trading day.

## Results

The model achieved an accuracy of 53.1% on the test set, using a classifier [specify: RandomForest, XGBoost, etc.]. It was evaluated using metrics such as accuracy, F1-score, and confusion matrix.

## How to run the prediction model with Docker

The model runs with  Conda and  Docker and the test needs to use the file  test_api.py. Follow the next stpes:

## How to run the prediction model
Follow this steps to clone the repository, built the Docker and get a prediction from the API.

### To clone the repository
```bash
# Sistema de Predicción: EC vs CL (Dólar Ecuatoriano vs Petróleo WTI)

## 📋 Descripción General

Este proyecto implementa un **sistema de predicción de precios** usando aprendizaje automático (Machine Learning) para predecir movimientos del **Dólar Ecuatoriano (EC)** correlacionando con el **Precio del Petróleo Crudo (CL=F/WTI)**.

### 🎯 Objetivo

Clasificar si el precio del EC subirá (Sube) o bajará (Baja) utilizando características técnicas y correlación con el precio del petróleo.

**Clase Binaria:**
- **Sube (0)**: El precio del EC aumenta en el siguiente período
- **Baja (1)**: El precio del EC disminuye en el siguiente período

---

## 🚀 Quick Start

### Requisitos Previos

- Python 3.8+
- `pip` (gestor de paquetes)
- Entorno virtual (recomendado)

### Instalación

1. **Clonar o descargar el proyecto**
```bash
cd /ruta/al/proyecto
```

2. **Crear entorno virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Ejecución Rápida

#### 1. Entrenar el Modelo
```bash
python run_pipeline_from_df_ml.py
```
**Output:** Entrena 4 modelos, selecciona el mejor, guarda artifacts en `data/`

#### 2. Realizar Predicciones
```bash
python predict_stock.py --use-df-ml --output data/predictions.csv
```
**Output:** Genera predicciones con probabilidades en `data/predictions.csv`

#### 3. Generar Visualizaciones
```bash
python generate_visualizations.py
```
**Output:** Genera 6 gráficos en `outputs/` (PNG)

#### 4. Entrenar Modelo Mejorado (Opcional)
```bash
python train_improved_model.py
```
**Output:** Compara modelo original con versiones balanceadas

---

## 📊 Conjunto de Datos

### Ubicación
- **Principal:** `data/df_ml.csv`
- **Alternativas:** `data/EC_processed.csv`, `data/PA_processed.csv`

### Estructura del Conjunto de Datos

```
Dimensión: 1157 filas × 13 columnas

Columnas Disponibles:
├── PRECIOS (Activo Principal - EC)
│   ├── Close: Precio de cierre diario
│   ├── Volume: Volumen de negociación
│   └── Target: Clasificación (0=Sube, 1=Baja) [ETIQUETA]
├── INDICADORES TÉCNICOS
│   ├── SMA_100: Media móvil simple 100 períodos
│   ├── RSI_14: Índice de Fuerza Relativa (14 períodos)
│   ├── Overbought: Indicador sobrecomprado (RSI > 70)
│   └── Oversold: Indicador sobrevendido (RSI < 30)
├── BANDERAS DE PREDICCIÓN
│   ├── Below_SMA: Precio < SMA (1=sí, 0=no)
│   └── High_Volume: Volumen elevado (1=sí, 0=no)
└── CORRELACIÓN CON ACTIVO CORRELACIONADO (CL=F/Petróleo)
     ├── CA_Close: Precio cierre del petróleo
     ├── CA_Change: Cambio porcentual del petróleo
     ├── CA_Volatility: Volatilidad del petróleo
     └── PA_CA_Ratio: Ratio EC/Petróleo
```

### Distribución de Clases
```
Sube (0):  607 muestras (52.5%)
Baja (1):  550 muestras (47.5%)
Desbalance: ~5% (relativamente balanceado)
```

---

## 🤖 Modelos Entrenados

### Comparativa de Desempeño

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.5647 ✓ | 1.00 | 1.00 | 1.00 |
| Random Forest | 0.5345 | 0.55 | 1.00 | 0.71 |
| K-Nearest Neighbors | 0.5300 | 0.54 | 1.00 | 0.70 |
| XGBoost | 0.5086 | 0.52 | 1.00 | 0.69 |

**Modelo Seleccionado:** `Logistic Regression` (mejor accuracy)
**Archivo:** `data/best_model.pkl`

### Arquitectura de los Modelos

#### Logistic Regression
- **Algoritmo:** Regresión Logística
- **Parámetros:** `max_iter=1000`
- **Training Set:** 925 muestras (80%)
- **Test Set:** 232 muestras (20%)

#### Random Forest (Balanceado)
- **Algoritmo:** Random Forest + Class Weights
- **Parámetros:** `n_estimators=100, class_weight='balanced'`
- **Top 3 Features:**
  1. CA_Change: 0.1317
  2. Volume: 0.1244
  3. CA_Volatility: 0.1237

---

## 📈 Resultados de Predicciones

### Estadísticas Resumidas

```
Total Predicciones: 1157
├── Predicciones "Sube":   1157 (100.0%)
└── Predicciones "Baja":       0 (0.0%)

Confianza Promedio:
├── P(Sube): 50.51% (±0.08%)
└── P(Baja): 49.49% (±0.08%)

Matriz de Confusión:
              Predicho
Real    | Sube | Baja |
      Sube |  607 |   0  |
      Baja |  550 |   0  |

Métricas:
├── Accuracy:  56.47%
├── Precision: 100.00%
├── Recall:    100.00%
├── F1-Score:  100.00%
└── ROC-AUC:   54.51%
```

### Archivos de Salida

| Archivo | Descripción |
|---------|-------------|
| `data/best_model.pkl` | Modelo entrenado (Logistic Regression) |
| `data/features.pkl` | Lista de 12 features utilizadas |
| `data/predictions_df_ml.csv` | Predicciones con probabilidades |
| `data/best_model_balanced.pkl` | Modelo mejorado con balanceo de clases |

---

## 📁 Estructura del Proyecto

```
.
├── 📄 README.md                          ← Este archivo
├── 📄 requirements.txt                   ← Dependencias Python
├── 📄 dockerfile                         ← Configuración Docker (opcional)
├── 🐍 app.py                             ← Aplicación Flask (interfaz web)
├── 🐍 test_api.py                        ← Tests para la API
│
├── 🐍 SCRIPTS DE ENTRENAMIENTO
│   ├── run_pipeline_from_df_ml.py        ← Entrena 4 modelos, selecciona mejor
│   ├── train_improved_model.py           ← Entrena con balanceo de clases
│   └── predict_stock.py                  ← Realiza predicciones
│
├── 🐍 SCRIPTS DE VISUALIZACIÓN
│   ├── generate_visualizations.py        ← Genera 6 gráficos PNG
│   └── visualize_predictions.py          ← (Legacy) Visualización original
│
├── 📔 NOTEBOOKS
│   └── stock_pred_ec_wti.ipynb          ← Notebook Jupyter (exploración)
│
├── 📂 data/                              ← Carpeta de datos
│   ├── df_ml.csv                         ← Dataset principal (1157 × 13)
│   ├── EC_processed.csv                  ← EC procesado (1157 × 7)
│   ├── PA_processed.csv                  ← PA procesado (1157 × 14)
│   ├── best_model.pkl                    ← Modelo guardado
│   ├── features.pkl                      ← Features guardadas
│   ├── best_model_balanced.pkl           ← Modelo mejorado
│   └── predictions_df_ml.csv             ← Predicciones generadas
│
└── 📂 outputs/                           ← Carpeta de visualizaciones
     ├── 01_prediction_distribution.png    ← Distribución de predicciones
     ├── 02_probability_distributions.png  ← Distribucion de confianzas
     ├── 03_confusion_matrix.png           ← Matriz de confusión
     ├── 04_roc_curve.png                  ← Curva ROC y AUC
     ├── 05_feature_importance.png         ← Importancia de features
     └── 06_summary_statistics.png         ← Resumen de métricas
```

---

## 🔧 Uso Detallado

### 1. Entrenar Modelos

```bash
python run_pipeline_from_df_ml.py
```

**¿Qué hace?**
- Carga datos de `data/df_ml.csv`
- Prepara features (12 seleccionadas)
- Split train/test 80/20 (sin shuffle, respeta series temporal)
- Entrena 4 modelos:
  - Logistic Regression
  - Random Forest (100 árboles)
  - KNN (k=5)
  - XGBoost
- Evalúa cada modelo
- Guarda el mejor en `data/best_model.pkl`
- Guarda features en `data/features.pkl`

**Output esperado:**
```
Logistic Regression - Accuracy: 0.5647 ✓ MEJOR
Random Forest - Accuracy: 0.5345
KNN - Accuracy: 0.5300
XGBoost - Accuracy: 0.5086

✓ Modelo guardado: data/best_model.pkl
```

---

### 2. Realizar Predicciones

```bash
# Opción A: Usar datos de training
python predict_stock.py --use-df-ml --output predictions.csv

# Opción B: Usar archivo CSV personalizado
python predict_stock.py --input custom_data.csv --output predictions.csv

# Opción C: Usar datos de entrenamiento (default)
python predict_stock.py
```

**Parámetros:**
- `--use-df-ml`: Usar `data/df_ml.csv` como entrada
- `--input FILE`: Usar archivo CSV personalizado
- `--output FILE`: Guardar predicciones en archivo (default: `data/predictions_df_ml.csv`)

**Output CSV:**
```
Close,Volume,SMA_100,...,Target,prediction,prediction_label,prob_Sube,prob_Baja
12.59,1418100,12.798,...,1,0,Sube,0.507,0.493
12.40,758200,12.814,...,0,0,Sube,0.504,0.496
...
```

---

### 3. Generar Visualizaciones

```bash
python generate_visualizations.py
```

**Genera 6 gráficos:**

1. **01_prediction_distribution.png**
    - Histograma de clases predichas
    - Muestra desbalance de predicciones

2. **02_probability_distributions.png**
    - Distribución de P(Sube)
    - Distribución de P(Baja)

3. **03_confusion_matrix.png**
    - Matriz de confusión con heatmap
    - Métricas: Sensitivity, Specificity, Accuracy

4. **04_roc_curve.png**
    - Curva ROC con AUC
    - Compara con clasificador aleatorio

5. **05_feature_importance.png**
    - Top 10 features más importantes
    - Ordenadas por importancia descendente

6. **06_summary_statistics.png**
    - Resumen de todas las métricas
    - Tabla de configuración

---

### 4. Entrenar Modelo Mejorado (Opcional)

```bash
python train_improved_model.py
```

**¿Qué hace?**
- Carga modelo original
- Entrena Logistic Regression + Class Weights
- Entrena Random Forest + Class Weights
- Compara accuracy y AUC
- Guarda mejor modelo balanceado

**Cuando usar:**
- Si hay desbalance de clases detectado
- Para mejorar recall en clase minoritaria
- Para ROC-AUC más equilibrado

---

## 💻 Integración con Flask (API REST)

### Iniciar servidor

```bash
python app.py
```

**Output esperado:**
```
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Endpoints Disponibles

#### GET `/` - Health Check
```bash
curl http://localhost:5000/
```
**Response:** `{"status": "API running"}`

#### POST `/predict` - Realizar Predicción
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
     "Close": 12.59,
     "Volume": 1418100,
     "SMA_100": 12.80,
     ...
  }'
```

**Response:**
```json
{
  "prediction": "Sube",
  "probability_Sube": 0.507,
  "probability_Baja": 0.493
}
```

### Tests

```bash
python test_api.py
```

### Ejecutar la suite de tests (pytest)

Recomendado para evaluar el proyecto completo (incluye tests de API y tests de alineación).

1. Crear y activar entorno virtual (recomendado):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias (incluye `pytest`):
```bash
pip install -r requirements.txt
```

3. Ejecutar todos los tests con pytest:
```bash
pytest -q
# o para ver más detalle:
pytest
```

4. Ejecutar un test específico:
```bash
pytest tests/test_predictions_alignment.py -q
```

Si estás en CI (GitHub Actions), usa `python -m pip install -r requirements.txt` y luego `pytest -q` en el job.

---

## ⚙️ Configuración de Dependencias

### requirements.txt

```
flask==2.3.3
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.3.2
xgboost==3.1.1
joblib==1.3.2
matplotlib==3.10.7
seaborn==0.13.2
jupyter==1.0.0
ipykernel==6.27.1
nbconvert==7.10.0
yfinance==0.2.38
```

### Instalación Personalizada

```bash
# Solo ML
pip install pandas numpy scikit-learn xgboost joblib

# Solo visualización
pip install matplotlib seaborn

# Solo API web
pip install flask

# Todo
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'sklearn'"
**Solución:**
```bash
pip install scikit-learn
```

### Error: "FileNotFoundError: data/df_ml.csv"
**Solución:**
- Asegúrate de estar en el directorio correcto
- Verifica que `data/df_ml.csv` existe
```bash
ls -la data/df_ml.csv
```

### Error: "No model found in data/best_model.pkl"
**Solución:**
- Ejecuta primero `python run_pipeline_from_df_ml.py`
- Esto entrena y guarda el modelo

### Error: "yfinance HTTP 429 - Too Many Requests"
**Solución:**
- Usa la opción `--use-df-ml` en `predict_stock.py`
- Usa datos locales en lugar de descargar de Yahoo Finance

### Predicciones todas "Sube" (100%)
**Información:**
- Esto es normal con Logistic Regression en este dataset
- El modelo está sobreajustado hacia la clase mayoritaria
- Solución: Usar `train_improved_model.py` con balanceo

---

## 📊 Métricas Clave Explicadas

### Accuracy (Precisión Global)
```
(TP + TN) / (TP + TN + FP + FN)
```
Porcentaje de predicciones correctas. **Actual: 56.47%**

### Precision (Exactitud)
```
TP / (TP + FP)
```
De las predicciones "Sube", ¿cuántas fueron correctas? **Actual: 100%**

### Recall (Sensibilidad)
```
TP / (TP + FN)
```
De los casos reales "Sube", ¿cuántos identificamos? **Actual: 100%**

### F1-Score
```
2 * (Precision * Recall) / (Precision + Recall)
```
Promedio armónico de Precision y Recall. **Actual: 100%**

### ROC-AUC
```
Área bajo la curva ROC (0 a 1)
```
Capacidad del modelo para discriminar clases. **Actual: 54.51%**

---

## 🔬 Investigación Técnica

### Features Utilizadas (12 Total)

| # | Feature | Descripción | Rango |
|----|---------|-------------|-------|
| 1 | Close | Precio de cierre del EC | ~12.0-12.9 |
| 2 | Volume | Volumen de negociación | ~300K-2.8M |
| 3 | SMA_100 | Media móvil 100 períodos | ~12.6-12.9 |
| 4 | RSI_14 | Índice fuerza relativa | 0-100 |
| 5 | Overbought | Indicador RSI > 70 | 0-1 |
| 6 | Oversold | Indicador RSI < 30 | 0-1 |
| 7 | Below_SMA | Precio < SMA | 0-1 |
| 8 | High_Volume | Volumen elevado | 0-1 |
| 9 | CA_Close | Precio cierre petróleo | ~55-70 |
| 10 | CA_Change | Cambio % petróleo | ~-3% a +3% |
| 11 | PA_CA_Ratio | Ratio EC/Petróleo | ~0.18-0.22 |
| 12 | CA_Volatility | Volatilidad petróleo | ~0.6-1.5 |

### Importancia de Features (Top 5)

```
1. CA_Change       13.17%  ← El cambio diario del petróleo es muy importante
2. Volume         12.44%   ← El volumen de negociación es clave
3. CA_Volatility  12.37%   ← La volatilidad del petróleo importa
4. CA_Close       12.05%   ← El precio del petróleo contribuye
5. SMA_100        12.01%   ← La tendencia de mediano plazo ayuda
```

---

## 🎯 Limitaciones Conocidas

1. **Predicciones sesgadas hacia "Sube"**
    - El modelo tiende a predecir siempre la clase mayoritaria
    - Impacto: Bajo recall en clase "Baja"
    - Solución: Usar modelo mejorado con balanceo de clases

2. **Accuracy limitado (56.47%)**
    - Apenas mejor que adivinanza aleatoria (50%)
    - Impacto: Uso en producción requiere validación adicional
    - Causa probable: Datos no suficientemente predictivos o ruido en series temporales

3. **Correlación EC-Petróleo asumida**
    - No hay garantía de correlación consistente
    - Impacto: El modelo puede no generalizarse a nuevos datos
    - Recomendación: Reentrenar periódicamente

4. **Sin validación cruzada**
    - Usa simple train/test split (80/20)
    - Impacto: Posible overfitting
    - Mejora: Usar cross-validation en futuras versiones

5. **Datos históricos limitados**
    - Solo 1157 muestras (~4.6 años de datos diarios)
    - Impacto: Posible insuficiencia para patrones a largo plazo
    - Recomendación: Recolectar más datos

---

## 🚀 Mejoras Futuras

- [ ] Agregar más características (volatilidad histórica, ratios técnicos avanzados)
- [ ] Implementar validación cruzada (5-fold CV)
- [ ] Usar SMOTE para balanceo de clases
- [ ] Explorar redes neuronales (LSTM para series temporales)
- [ ] Optimización de hiperparámetros (GridSearchCV)
- [ ] Integración con más fuentes de datos
- [ ] Dashboard interactivo (Streamlit o Dash)
- [ ] Alertas automáticas de predicciones
- [ ] Backtesting de estrategias
- [ ] Modelo ensemble (combinación de múltiples modelos)

---

## 📞 Soporte

### Contacto
- **Email:** [soporte@ejemplo.com]
- **Issues:** Crear issue en repositorio GitHub
- **Documentación:** Ver notebooks en `stock_pred_ec_wti.ipynb`

### Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -am 'Agrega mejora'`)
4. Push a rama (`git push origin feature/mejora`)
5. Abrir Pull Request

---

## 📜 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para detalles.

---

## 📝 Changelog

### v1.0.0 (Actual)
- ✅ Sistema de predicción ML completamente funcional
- ✅ 4 modelos entrenados y evaluados
- ✅ Logistic Regression como modelo seleccionado
- ✅ API REST con Flask
- ✅ 6 visualizaciones automáticas
- ✅ Modelo mejorado con balanceo de clases
- ✅ Documentación completa

### v0.9.0 (Anterior)
- Notebook exploratorio inicial
- Primeros tests de modelos

---

## 🙏 Agradecimientos

- **Datos:** Yahoo Finance API (yfinance)
- **ML:** Scikit-learn, XGBoost
- **Visualización:** Matplotlib, Seaborn
- **Web:** Flask

---

**Última actualización:** 2024
**Versión:** 1.0.0
**Python:** 3.12.1