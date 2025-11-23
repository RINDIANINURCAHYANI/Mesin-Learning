# ============================================
# Klasifikasi Risiko Penyakit Jantung
# Metode: Decision Tree
# Dataset: Heart Disease UCI (heart_disease_uci.csv)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# -----------------------------
# Langkah 1: Load Data
# -----------------------------
# Langkah 1: Load Data
df = pd.read_csv(r"C:\Users\rindi\OneDrive\Dokumen\PENYAKIT JANTUNG\heart_disease_uci.csv")

print("=== 5 Data Teratas ===")
print(df.head())
print("\n=== Info Data ===")
print(df.info())

# -----------------------------
# Langkah 2: Preprocessing
# -----------------------------

# 1) Buat target biner "risk"
# num = 0  -> tidak ada penyakit (risiko rendah)
# num > 0 -> ada penyakit (berisiko)
df["risk"] = (df["num"] > 0).astype(int)

# 2) Hapus kolom yang tidak dipakai
# (num digantikan oleh risk)
cols_to_drop = []
for c in ["id", "dataset", "num"]:
    if c in df.columns:
        cols_to_drop.append(c)

df = df.drop(columns=cols_to_drop)

# 3) Tentukan fitur numerik & kategorikal (sesuai struktur umum UCI)
num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# 4) Tangani missing values
# - Numerik: median
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# - Kategorikal: modus
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].mode()[0])

# 5) One-hot encoding untuk kategorikal
existing_cat_cols = [c for c in cat_cols if c in df.columns]
df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

print("\n=== Data Setelah Preprocessing ===")
print(df.head())

# -----------------------------
# Langkah 3: Train-Test Split
# -----------------------------
X = df.drop(columns=["risk"])
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -----------------------------
# Langkah 4: Training Decision Tree
# -----------------------------
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Langkah 5: Evaluasi Model
# -----------------------------
predictions = model.predict(X_test)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=["Risiko Rendah", "Berisiko"]))

# -----------------------------
# Langkah 6: Visualisasi Pohon Keputusan
# -----------------------------
plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Risiko Rendah", "Berisiko"],
    filled=True,
    rounded=True
)
plt.title("Visualisasi Pohon Keputusan - Klasifikasi Risiko Penyakit Jantung (max_depth=3)")
plt.show()

# -----------------------------
# Langkah 7: Heatmap Confusion Matrix
# -----------------------------
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Risiko Rendah", "Berisiko"],
    yticklabels=["Risiko Rendah", "Berisiko"]
)
plt.xlabel("Label Prediksi")
plt.ylabel("Label Sebenarnya")
plt.title("Heatmap Confusion Matrix")
plt.show()

# -----------------------------
# Langkah 8: Feature Importance
# -----------------------------
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Fitur": X_train.columns,
    "Pentingnya": importances
}).sort_values(by="Pentingnya", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Pentingnya", y="Fitur", data=feature_importance_df)
plt.title("Peringkat Pentingnya Fitur (Feature Importance)")
plt.xlabel("Nilai Pentingnya")
plt.ylabel("Fitur")
plt.show()

# -----------------------------
# Langkah 9: ROC Curve & AUC
# -----------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Tebakan Acak")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - Klasifikasi Risiko Penyakit Jantung")
plt.legend()
plt.grid(True)
plt.show()
