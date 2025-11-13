📈 Stock Price Movement Prediction: Ecopetrol (EC) vs. WTI Oil

This academic project applies Machine Learning techniques to predict whether Ecopetrol S.A. (EC) stock price will go up or down the next day, correlating historical data with the WTI Crude Oil (CL=F) price. The system provides a trained model, a functional REST API, and a Dockerized environment.

🎯 Core Objective

To develop a binary classification system (0 = Up, 1 = Down) for EC stock price movement, based on technical indicators and oil price correlation/volatility.

🚀 Quick Start Guide
Follow these steps to set up the system, train the model, and generate predictions.

1. Prerequisites and Installation

   1. Clone the repository:
   ```bash
   git clone https://github.com/Rasosa31/123456.git
   cd 123456
   ```

   2. Create and activate a virtual environment:

   ```Bash
   python -m venv .venv
   source .venv/bin/activate # For Windows: .venv\Scripts\activate
   ```

   3. Install dependencies (includes scikit-learn, Flask, Pytest):

   ```Bash
   pip install -r requirements.txt

2. Quick Script Execution

 ```bash

Script             Description                                                        Command

Train Model        Trains 4 models and saves the best one (Logistic Regression)       python run_pipeline_from_df_ml.py
                   in data/best_model.pkl.

Predict            Generates predictions on the test dataset.                         python predict_stock.py --use-df-ml

Visualize          Generates 6 PNG charts (Confusion Matrix, ROC, etc.) in the        python generate_visualizations.py
                   outputs/ folder.
```

💻 REST API Integration (Flask)

The model is exposed via a web API to perform real-time predictions.

Start the Server

```Bash
python app.py
Expected Output: * Running on http://127.0.0.1:5000
```

Available Endpoints

1. GET / - Health Check

```Bash
curl http://localhost:5000/
Response: {"status": "API running"}
```

2. POST /predict - Make Prediction

Input (JSON):

```Bash
Feature	Description	Example
Close	EC closing price	12.59
Volume	EC trading volume	1418100
SMA_100	Simple Moving Average (100-period)	12.80
RSI_14	Relative Strength Index (14-period)	56.0
CA_Close	WTI Oil closing price	46.32
```
```Bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Close": 12.59,
    "Volume": 1418100,
    "SMA_100": 12.80,
    "RSI_14": 56.0,
    "CA_Close": 46.32
}'
```

Response (JSON):
```Bash
JSON
{
  "prediction": "Sube" ,
  "probability_Sube": 0.507,
  "probability_Baja": 0.493
}
```
Tests

Run the test suite for API functionality and data integrity (requires the model to be saved).

```Bash
pytest -q
```

🤖 Model and Performance (Logistic Regression)
The selected model was Logistic Regression due to its initial highest Accuracy on the test set.

Performance Comparison (Test Set)

```Bash
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression (Selected)	0.5647	1.00	1.00	1.00
Random Forest	0.5345	0.55	1.00	0.71
K-Nearest Neighbors	0.5300	0.54	1.00	0.70
XGBoost	0.5086	0.52	1.00	0.69
```

Confusion Matrix and Key Metrics
```bash
Metric	Value	Observation
Accuracy	56.47%	Slightly better than random chance (50%).
Precision	100.00%	Of the "Up" predictions, 100% were correct.
Recall	100.00%	Of the actual "Up" cases, 100% were identified.
ROC-AUC	54.51%	Low ability to discriminate between classes.
```

Important Note: The high Precision and Recall (100%) are due to an over-adjustment towards the majority class (Up). The confusion matrix shows the model predicts Up 100% of the time, indicating a strong bias.

🔬 Features Used (Top 5 Importance)

The model uses 12 features, including price, volume, and 7 technical indicators for EC, plus 4 correlation features related to WTI Oil.

```bash
Feature	Importance	Data Type
1. CA_Change	13.17%	WTI Oil daily percentage change.
2. Volume	12.44%	EC trading volume.
3. CA_Volatility	12.37%	WTI Oil volatility.
4. CA_Close	12.05%	WTI Oil closing price.
5. SMA_100	12.01%	Trend (Simple Moving Average 100-period).
```

🚨 Known Limitations

Bias towards "Up" Prediction (Critical Limit): The current model (Logistic Regression) tends to always predict the majority class (Up). Solution: Run python train_improved_model.py to train models with class balancing.

```bash
Limited Accuracy (56.47%): Low performance requires caution in a production environment.
No Cross-Validation: A simple train/test split (80/20) was used, increasing the risk of overfitting.
Limited Historical Data: Only 1157 samples (approx. 4.6 years)
```
🛠️ Project Structure
```bash
.
├── 📄 README.md
├── 📄 requirements.txt                   ← Dependencies
├── 🐍 app.py                             ← Flask Application (REST API)
├── 🐍 test_api.py                        ← Tests for the API
│
├── 🐍 TRAINING SCRIPTS
│   ├── run_pipeline_from_df_ml.py        ← Trains 4 models
│   └── train_improved_model.py           ← Trains with class balancing
│
├── 📂 data/                              ← Dataset and Artifacts
│   ├── df_ml.csv                         ← Primary Dataset (1157 × 13)
│   ├── best_model.pkl                    ← Saved Model (LogReg)
│   └── predictions_df_ml.csv             ← Generated Predictions
│
└── 📂 outputs/                           ← Charts (Confusion Matrix, ROC, etc.)
```
🐛 Troubleshooting (Common Issues)

```bash
Error	Solution
ModuleNotFoundError: No module named 'sklearn'	pip install scikit-learn or pip install -r requirements.txt
FileNotFoundError: data/df_ml.csv	Verify you are in the root directory.
No model found in data/best_model.pkl	Run python run_pipeline_from_df_ml.py first.
100% "Up" Predictions	This is normal due to bias. Run python train_improved_model.py to use a balanced model.
```

