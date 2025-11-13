# Stock Price Movement Prediction with Machine Learning

[![CI](https://github.com/Rasosa31/123456/actions/workflows/ci.yml/badge.svg)](https://github.com/Rasosa31/123456/actions)
 
CI: tests automáticos (pytest) — se ejecuta en cada push y pull request a `main`.
Ver el historial y el estado de las ejecuciones en: https://github.com/Rasosa31/123456/actions

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
# Prediction System: EC vs CL (Ecopetrol S.A vs Oil WTI)

## 📋 General Description

This project implements a **price prediction system** using machine learning (Machine Learning) to predict movements of **Ecopetrol S.A (EC)** correlating with the **Price of Crude Oil (CL=F/WTI)**.

### 🎯 Objetive

Classify if the price of the EC will go up (up) or down (down) using technical characteristics and correlation with the price of oil.
**Clase Binaria:**
- **Goes up (0)**: The price of the EC increases in the following period
- **Goes down (1)**: The price of the EC decreases in the following period

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- `pip` (Package manager)
-Virtual environment (recommended)

### Instalación

1. **Clone or download the project**
```bash
cd /ruta/al/proyecto
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Fast Execution

#### 1. Entrenar el Modelo
```bash
python run_pipeline_from_df_ml.py
```
**Output:** Train 4 models, select the best, save artifacts in `data/`

#### 2. Make Predictions
```bash
python predict_stock.py --use-df-ml --output data/predictions.csv
```
**Output:** Generate predictions with probabilities in `data/predictions.csv`

#### 3. Generate Visualizations
```bash
python generate_visualizations.py
```
**Output:** Generates 6 charts in `outputs/` (PNG)

#### 4. Train Improved Model (Optional)
```bash
python train_improved_model.py
```
**Output:** Compare original model with balanced versions

---

## 📊 Data Set

### Location
- **Principal:** `data/df_ml.csv`
- **Alternatives:** `data/EC_processed.csv`, `data/PA_processed.csv`

### Data Set Structure

```

Available Columns:
├── PRICES (Main Asset - EC)
│   ├── Close: Daily closing price
│   ├── Volume: Volume of negotiation
│   └── Target: Classification (0=Up, 1=down) [TAG]
├── TECHNICAL INDICATORS
│   ├── SMA_100: Simple moving average 100 periods
│   ├── RSI_14: Relative Strength Index (14 periods)
│   ├── Overbought: Indicator overbought (RSI > 70)
│   └── Oversold: Indicator oversell (RSI < 30)
├── PREDICTION FLAGS
│   ├── Below_SMA: Price < SMA (1=yes, 0=no)
│   └── High_Volume: High volume(1=yes, 0=no)
└── CORRELACIÓN CON ACTIVO CORRELACIONADO (CL=F/oil)
     ├── CA_Close: Oil closing price
     ├── CA_Change: Percentage change in oil
     ├── CA_Volatility: Oil volatility
     └── PA_CA_Ratio: Ratio EC/oil
```

### Distribution of Classes
```
Rise (0): 607 samples (52.5%)
Low (1): 550 samples (47.5%)
Imbalance: ~5% (relatively balanced)

```

---

## 🤖 Trained Models

### Performance Comparison

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.5647 ✓ | 1.00 | 1.00 | 1.00 |
| Random Forest | 0.5345 | 0.55 | 1.00 | 0.71 |
| K-Nearest Neighbors | 0.5300 | 0.54 | 1.00 | 0.70 |
| XGBoost | 0.5086 | 0.52 | 1.00 | 0.69 |

**Selected Model:** `Logistic Regression` (mejor accuracy)
**File:** `data/best_model.pkl`

### Architecture of the Models

#### Logistic Regression
- **Algoritmo:** Regresión Logística
- **Parámetros:** `max_iter=1000`
- **Training Set:** 925 muestras (80%)
- **Test Set:** 232 muestras (20%)

#### Random Forest (Balanceado)
- **Algoritmo:** Random Forest + Class Weights
- **Parameters:** `n_estimators=100, class_weight='balanced'`
- **Top 3 Features:**
  1. CA_Change: 0.1317
  2. Volume: 0.1244
  3. CA_Volatility: 0.1237

---

## 📈 Predictions Results

### Summary Statistics

```
Total Predictions: 1157
├── Predictions "Goes up":   1157 (100.0%)
└── Predictions "Goes down":       0 (0.0%)

Average Confidence:
├── P(goes up): 50.51% (±0.08%)
└── P(goes down): 49.49% (±0.08%)

Confusion: Matriz  
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

### Output Files

| File       | Description |
|---------|-------------|
| `data/best_model.pkl` | Trained model (Logistic Regression) |
| `data/features.pkl` | List of 12 features used|
| `data/predictions_df_ml.csv` | Predictions with probabilities |
| `data/best_model_balanced.pkl` | Improved model with class balance|

---

## 📁 Project Structure

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

## 🔧 Detailed Use

### 1. Train Models

```bash
python run_pipeline_from_df_ml.py
```

**What does it do? **

- Load data from `data/df_ml.csv`
- Prepare features (12 selected)
- Split train/test 80/20 (without shuffle, respects temporary series)
- Train 4 models:
- Logistic Regression
- Random Forest (100 trees)
- KNN (k=5)
- XGBoost
- Evaluate each model
- Save the best in `data/best_model.pkl`
- Save features in `data/features.pkl`

**Output awaited:**
```
Logistic Regression - Accuracy: 0.5647 ✓ MEJOR
Random Forest - Accuracy: 0.5345
KNN - Accuracy: 0.5300
XGBoost - Accuracy: 0.5086

✓ Model saved: data/best_model.pkl
```

---

### 2. Make Predictions

```bash
# Option A: Use training data
Python predict_stock.py --use-df-ml --output predictions.csv

# Option B: Use custom CSV file
Python predict_stock.py --input custom_data.csv --output predictions.csv

# Option C: Use training data (default)
Python predict_stock.py

```

**Parámetros:**
- `--use-df-ml`: Use `data/df_ml.csv` As an entrance
- `--input FILE`: Use custom CSV file
- `--output FILE`: Save predictions in file (default: `data/predictions_df_ml.csv`)

**Output CSV:**
```
Close,Volume,SMA_100,...,Target,prediction,prediction_label,prob_goes up,prob_goes down
12.59,1418100,12.798,...,1,0,Sube,0.507,0.493
12.40,758200,12.814,...,0,0,Sube,0.504,0.496
...
```

---

### 3. Generate Visualizations

```bash
python generate_visualizations.py
```

**Output 6 graphs:**

1. **01_prediction_distribution.png**
- Histogram of predicted classes
- Sample imbalance of predictions

2. **02_probability_distributions.png**
    - Distribution de P(Sube)
    - Distribution de P(Baja)

3. **03_confusion_matrix.png**
    - Matrix of confusion withheatmap
    - Métricas: Sensitivity, Specificity, Accuracy

4. **04_roc_curve.png**
    - Curva ROC con AUC
    -Compare with random classifier

5. **05_feature_importance.png**
6. 
- Top 10 most important features
- Ordered by descending importance

6. **06_summary_statistics.png**
- Summary of all metrics
- Configuration table
---

### 4. Train Improved Model (Optional)

```bash
python train_improved_model.py
```

**¿What does it do??**

-Original model load
- Train Logistic Regression + Class Weights
- Train Random Forest + Class Weights
- Compare accuracy and AUC
- Save better balanced model
- 
**When to use:**

- If there is a class imbalance detected
- To improve recall in minority class
- For more balanced ROC-AUC
---

## 💻 Integration with Flask (API REST)

### Start server

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

### Run the test suite (pytest)

Recommended to evaluate the entire project (includes API tests and alignment tests).

1. Create and activate virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (includes `pytest`):
```bash
pip install -r requirements.txt
```

3. Run all tests with pytest:
```bash
pytest -q

# Or to see more details:
pytest
```

4. Run a specific test:
```bash
pytest tests/test_predictions_alignment.py -q
```

If you are in CI (GitHub Actions), use `python -m pip install -r requirements.txt` y luego `pytest -q` in the  job.

---

## ⚙️ Dependencies Configuration

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
# Only ML
pip install pandas numpy scikit-learn xgboost joblib

# Visualization only
pip install matplotlib seaborn

# Only API web
pip install flask

# All
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
**Solution:**
- Make sure you are in the correct directory
- Verify that `data/df_ml.csv` exists
```bash
ls -la data/df_ml.csv
```

### Error: "No model found in data/best_model.pkl"
**Solución:**
-Run `python run_pipeline_from_df_ml.py` first
- This trains and saves the model
- 
### Error: "yfinance HTTP 429 - Too Many Requests"
**Solución:**
-Use the option `--use-df-ml` in `predict_stock.py`
- Use local data instead of downloading from Yahoo Finance

### Predictions all "Goes up" (100%)

**Información:**
- This is normal with Logistic Regression in this dataset
- The model is overadjusted towards the majority class
- Solution: Use `train_improved_model.py` with balancing

---

## 📊 Key Metrics Explained

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
Of the real "Sube" cases, how many do we identify? **Current: 100%**

### F1-Score
```
2 * (Precision * Recall) / (Precision + Recall)
```
Harmonic average of Precision and Recall. **Current: 100%**

### ROC-AUC
```
Area under the curve ROC (0 a 1)
```
Ability of the model to discriminate classes. **Actual: 54.51%**

---

## 🔬 Investigación Técnica

### Features Used (12 Total)

| # | Feature | Description | Range |
|----|---------|-------------|-------|
| 1 | Close | EC closing price| ~12.0-12.9 |
| 2 | Volume | Trading volume| ~300K-2.8M |
| 3 | SMA_100 | Moving average 100 periods | ~12.6-12.9 |
| 4 | RSI_14 | Relative force index | 0-100 |
| 5 | Overbought | Indicator RSI > 70 | 0-1 |
| 6 | Oversold | Indicator RSI < 30 | 0-1 |
| 7 | Below_SMA | Price < SMA | 0-1 |
| 8 | High_Volume | High volume | 0-1 |
| 9 | CA_Close | Oil closing price | ~55-70 |
| 10 | CA_Change | Change % oil | ~-3% a +3% |
| 11 | PA_CA_Ratio | Ratio EC/Oil | ~0.18-0.22 |
| 12 | CA_Volatility | Oil volatility| ~0.6-1.5 |

### Importance of Features (Top 5)
```
1. CA_Change 13.17% ← The daily oil change is very important
2. Volume 12.44% ← Trading volume is key
3. CA_Volatility 12.37% ← Oil volatility matters
4. CA_Close 12.05% ← The price of oil contributes
5. SMA_100 12.01% ← Medium-term trend helps
```

---

## 🎯 Known Limitations

1. **Predictions biased towards "Sube"**

- The model tends to always predict the majority class
- Impact: Low recall in "Low" class
- Solution: Use improved model with class balancing
- 
2. **Accuracy limitado (56.47%)**
- Hardly better than random guessing (50%)
- Impact: Use in production requires additional validation
- Probable cause: Not sufficiently predictive data or noise in time series
3. **EC-Oil correlation assumed**

- There is no guarantee of consistent correlation
- Impact: The model may not be generalized to new data
- Recommendation: Re-train periodically

4. **No cross-validation**
- Use simple train/test split (80/20)
- Impact: Possible overfitting
- Improvement: Use cross-validation in future versions

5. **Limited historical data**
- Only 1157 samples (~4.6 years of daily data)
- Impact: Possible insufficiency for long-term patterns
- Recommendation: Collect more data

---

## 🚀 Future Improvements

- [ ] Add more features (historical volatility, advanced technical ratios)
- [ ] Implement cross validation (5-fold CV)
- [ ] Use SMOTE for class balancing
- [ ] Explore neural networks (LSTM for time series)
- [ ] Hyperparameter optimization (GridSearchCV)
- [ ] Integration with more data sources
- [ ] Interactive Dashboard (Streamlit or Dash)
- [ ] Automatic prediction alerts
- [ ] Backtesting of strategies
- [ ] Ensemble model (multiple model combination)
---

## 📝 Changelog

### v1.0.0 (Current)

- ✅ Fully functional ML prediction system
- ✅ 4 models trained and evaluated
- ✅ Logistic Regression as a selected model
- ✅ API REST with Flask
- ✅ 6 automatic views
- ✅ Improved model with class balancing
- ✅ Complete documentation

### v0.9.0 (Anterior)
- Notebook exploratorio inicial
- Primeros tests de modelos

---

## 🙏 Thanks

- **Data:** Yahoo Finance API (yfinance)
- **ML:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Web:** Flask

---

