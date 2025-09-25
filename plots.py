import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import plot_tree

#load
df = pd.read_excel("BlaBla.xlsx")
target_col = "A"
feature_cols = [c for c in df.columns if c != target_col]
if "UMUR_TAHUN" in feature_cols:
    df["UMUR_TAHUN"] = pd.to_numeric(df["UMUR_TAHUN"], errors="coerce")
feature_cols = [c for c in feature_cols if df[c].nunique() > 1]

X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
y = df[target_col].astype(int)

#80/20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = joblib.load("model.pkl")

#pohon
plt.figure(figsize=(14,10))
plot_tree(clf, feature_names=feature_cols,
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True, fontsize=8)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)

#confusion matrix
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(values_format='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
