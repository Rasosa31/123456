#!/usr/bin/env python3
"""
Script de visualización: Genera gráficos de predicciones y análisis del modelo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os

# Crear directorio de salida
os.makedirs('outputs', exist_ok=True)

print("="*70)
print("GENERANDO VISUALIZACIONES")
print("="*70)

# Cargar datos
print("\nCargando datos...")
df_ml = pd.read_csv('data/df_ml.csv').dropna()
predictions_df = pd.read_csv('data/predictions_df_ml.csv')
model = joblib.load('data/best_model.pkl')
features = joblib.load('data/features.pkl')

print(f"✓ Datos cargados: {df_ml.shape}")
print(f"✓ Predicciones: {predictions_df.shape}")

# Preparar datos
X = df_ml[features]
y_true = df_ml['Target']
y_pred_text = predictions_df['prediction_label']
# Convertir prediction_label a números (Sube=0, Baja=1)
y_pred = (y_pred_text == 'Baja').astype(int)
prob_sube = predictions_df['prob_Sube']
prob_baja = predictions_df['prob_Baja']

# Alineamiento entre ground-truth (df_ml) y predictions_df.
# En algunos casos el archivo de predicciones contiene más filas
# (p.ej. predicciones para un conjunto mayor). Intentamos unir
# por columnas clave redondeadas y, si falla, realizamos un
# truncamiento por índice como último recurso.
if len(df_ml) != len(predictions_df):
  print(f"! Atención: filas ground-truth={len(df_ml)} vs preds={len(predictions_df)}. Intentando alinear...")

  # Columnas candidatas para hacer el merge (redondeamos floats para robustez)
  merge_cols = []
  for c in ['Close', 'CA_Close']:
    if c in df_ml.columns and c in predictions_df.columns:
      df_ml[c + '_r'] = df_ml[c].round(3)
      predictions_df[c + '_r'] = predictions_df[c].round(3)
      merge_cols.append(c + '_r')

  # Volume como entero (si está presente)
  if 'Volume' in df_ml.columns and 'Volume' in predictions_df.columns:
    df_ml['Volume_r'] = df_ml['Volume'].astype('Int64')
    predictions_df['Volume_r'] = predictions_df['Volume'].astype('Int64')
    merge_cols.append('Volume_r')

  merged = None
  if merge_cols:
    try:
      merged = pd.merge(df_ml.reset_index(drop=True),
                predictions_df.reset_index(drop=True),
                left_on=merge_cols,
                right_on=merge_cols,
                how='inner',
                suffixes=('_true', '_pred'))
      print(f"✓ Merge realizado usando columnas: {merge_cols}. Filas resultantes: {len(merged)}")
    except Exception as e:
      print(f"! Error intentando merge: {e}")
      merged = None

  # Si no encontramos coincidencias razonables, alineamos por índice truncando al mínimo común
  if merged is None or len(merged) == 0:
    min_len = min(len(df_ml), len(predictions_df))
    print(f"! Merge no produjo filas. Usando truncamiento por índice al mínimo común: {min_len} filas.")
    df_ml = df_ml.reset_index(drop=True).iloc[:min_len]
    predictions_df = predictions_df.reset_index(drop=True).iloc[:min_len]
    # Recalcular variables alineadas
    X = df_ml[features]
    y_true = df_ml['Target']
    y_pred_text = predictions_df['prediction_label']
    y_pred = (y_pred_text == 'Baja').astype(int)
    prob_sube = predictions_df['prob_Sube']
    prob_baja = predictions_df['prob_Baja']
  else:
    # Usar las filas del merge para métricas pareadas
    y_true = merged['Target'] if 'Target' in merged.columns else merged['Target_true']
    y_pred_text = merged['prediction_label']
    y_pred = (y_pred_text == 'Baja').astype(int)
    prob_sube = merged['prob_Sube']
    prob_baja = merged['prob_Baja']
    # Para plots que usan X/features, intentar reconstruir X desde merged cuando sea posible
    try:
      X = merged[[f for f in features if f in merged.columns]]
    except Exception:
      X = df_ml[features]

# ============================================================================
# 1. Distribución de predicciones
# ============================================================================
print("\n[1/6] Distribución de predicciones...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pred_counts = pd.Series(y_pred).value_counts().sort_index()
colors = ['#2ecc71', '#e74c3c']  # Verde para Sube, Rojo para Baja
bars = ax.bar(['Sube (0)', 'Baja (1)'], [pred_counts.get(0, 0), pred_counts.get(1, 0)],
              color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Cantidad de Predicciones', fontsize=12, fontweight='bold')
ax.set_xlabel('Clase de Predicción', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Predicciones del Modelo', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Agregar valores en las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({100*height/len(y_pred):.1f}%)',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/01_prediction_distribution.png")

# ============================================================================
# 2. Distribución de probabilidades
# ============================================================================
print("[2/6] Distribución de probabilidades...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Probabilidad de Sube
ax1.hist(prob_sube, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Probabilidad', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax1.set_title('Distribución P(Sube)', fontsize=12, fontweight='bold')
ax1.axvline(prob_sube.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Media: {prob_sube.mean():.3f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Probabilidad de Baja
ax2.hist(prob_baja, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Probabilidad', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax2.set_title('Distribución P(Baja)', fontsize=12, fontweight='bold')
ax2.axvline(prob_baja.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Media: {prob_baja.mean():.3f}')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_probability_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/02_probability_distributions.png")

# ============================================================================
# 3. Matriz de confusión
# ============================================================================
print("[3/6] Matriz de confusión...")
fig, ax = plt.subplots(figsize=(8, 7))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=['Sube', 'Baja'], yticklabels=['Sube', 'Baja'],
            ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')

# Calcular métricas
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
acc = accuracy_score(y_true, y_pred)

textstr = f'Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\nAccuracy: {acc:.3f}'
ax.text(2.5, 0.5, textstr, transform=ax.transData, fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/03_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/03_confusion_matrix.png")

# ============================================================================
# 4. Curva ROC y AUC
# ============================================================================
print("[4/6] Curva ROC...")
fig, ax = plt.subplots(figsize=(8, 7))
try:
    fpr, tpr, thresholds = roc_curve(y_true, prob_sube)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Curva ROC', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
except Exception as e:
    ax.text(0.5, 0.5, f'Error generando ROC: {str(e)}', ha='center', va='center')

plt.tight_layout()
plt.savefig('outputs/04_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/04_roc_curve.png")

# ============================================================================
# 5. Importancia de features
# ============================================================================
print("[5/6] Importancia de features...")
fig, ax = plt.subplots(figsize=(10, 6))

# Obtener importancia del modelo
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
elif hasattr(model, 'coef_'):
    importances = np.abs(model.coef_[0])
else:
    importances = np.ones(len(features))

# Top 10 features
top_idx = np.argsort(importances)[-10:][::-1]
top_features = [features[i] for i in top_idx]
top_importances = importances[top_idx]

colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax.barh(range(len(top_features)), top_importances, color=colors_bar, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features, fontsize=10)
ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Features por Importancia', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Agregar valores
for i, (bar, val) in enumerate(zip(bars, top_importances)):
    ax.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/05_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/05_feature_importance.png")

# ============================================================================
# 6. Estadísticas resumidas
# ============================================================================
print("[6/6] Generando resumen de estadísticas...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Calcular métricas
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

metrics_text = f"""
RESUMEN DE ESTADÍSTICAS DEL MODELO

Datos Generales:
  • Total de muestras: {len(y_true):,}
  • Muestras de Sube: {(y_true == 0).sum():,} ({100*(y_true == 0).sum()/len(y_true):.1f}%)
  • Muestras de Baja: {(y_true == 1).sum():,} ({100*(y_true == 1).sum()/len(y_true):.1f}%)

Predicciones:
  • Predichas como Sube: {(y_pred == 0).sum():,} ({100*(y_pred == 0).sum()/len(y_pred):.1f}%)
  • Predichas como Baja: {(y_pred == 1).sum():,} ({100*(y_pred == 1).sum()/len(y_pred):.1f}%)

Métricas de Desempeño:
  • Accuracy: {accuracy:.4f}
  • Precision: {precision:.4f}
  • Recall: {recall:.4f}
  • F1-Score: {f1:.4f}

Matriz de Confusión:
  • Verdaderos Positivos (TP): {tp}
  • Falsos Positivos (FP): {fp}
  • Verdaderos Negativos (TN): {tn}
  • Falsos Negativos (FN): {fn}

Probabilidades:
  • P(Sube) - Media: {prob_sube.mean():.4f}, Std: {prob_sube.std():.4f}
  • P(Baja) - Media: {prob_baja.mean():.4f}, Std: {prob_baja.std():.4f}

Features Utilizadas: {len(features)}
  {', '.join(features[:5])}...
"""

ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('outputs/06_summary_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Guardado: outputs/06_summary_statistics.png")

print("\n" + "="*70)
print("✓ TODAS LAS VISUALIZACIONES COMPLETADAS")
print("="*70)
print("\nArchivos generados en outputs/:")
print("  1. 01_prediction_distribution.png")
print("  2. 02_probability_distributions.png")
print("  3. 03_confusion_matrix.png")
print("  4. 04_roc_curve.png")
print("  5. 05_feature_importance.png")
print("  6. 06_summary_statistics.png")
