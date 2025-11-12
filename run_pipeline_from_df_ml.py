#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print('Loading data/df_ml.csv')
df_ml = pd.read_csv('data/df_ml.csv')
print('shape:', df_ml.shape)

# define features if features.pkl exists
if os.path.exists('data/features.pkl'):
    try:
        features = joblib.load('data/features.pkl')
        print('Loaded features from data/features.pkl')
    except Exception:
        features = [c for c in df_ml.columns if c!='Target']
        print('Could not load features.pkl; using df_ml columns')
else:
    features = [c for c in df_ml.columns if c!='Target']
    print('Using inferred features from df_ml')

print('Features used:', features)

# drop na and ensure types
df_ml = df_ml.dropna().reset_index(drop=True)
print('After dropna shape:', df_ml.shape)

X = df_ml[features]
y = df_ml['Target']

if len(X)==0:
    print('No samples in X after dropna; aborting')
    raise SystemExit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f'Train: {len(X_train)} | Test: {len(X_test)}')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results={}
for name, model in models.items():
    print('\nTraining', name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name]=acc
    print(f"{name}: {acc:.3f}")

best_name = max(results, key=results.get)
best_model = models[best_name]
print('\nBest model:', best_name, results[best_name])

# detailed eval
y_pred = best_model.predict(X_test)
print('\nClassification report:')
print(classification_report(y_test, y_pred, target_names=['Sube','Baja']))

# save artifacts
os.makedirs('data', exist_ok=True)
joblib.dump(best_model, 'data/best_model.pkl')
joblib.dump(features, 'data/features.pkl')
print('Saved best_model and features to data/')
# También generamos predicciones para todo el dataset procesado (df_ml)
try:
    print('\nGenerando predicciones para todo data/df_ml.csv...')
    full_preds = best_model.predict(X)
    probs = None
    try:
        probs = best_model.predict_proba(X)
    except Exception:
        pass

    results_df = df_ml.copy()
    results_df['prediction'] = full_preds
    results_df['prediction_label'] = results_df['prediction'].map({0: 'Sube', 1: 'Baja'})
    if probs is not None:
        # Asegurarse del número correcto de columnas
        if probs.shape[1] >= 2:
            results_df['prob_Sube'] = probs[:, 0]
            results_df['prob_Baja'] = probs[:, 1]
        else:
            results_df['prob_0'] = probs[:, 0]

    out_path = 'data/predictions_df_ml.csv'
    results_df.to_csv(out_path, index=False)
    print(f'✓ Predicciones guardadas en: {out_path} (filas: {len(results_df)})')
except Exception as e:
    print('! No se pudieron generar predicciones completas:', e)
