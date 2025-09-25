import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#Load
df = pd.read_excel("BlaBla.xlsx")

#Target & fitur
target_col = "A"
feature_cols = [c for c in df.columns if c != target_col]

#Preprocessing
if "UMUR_TAHUN" in feature_cols:
    df["UMUR_TAHUN"] = pd.to_numeric(df["UMUR_TAHUN"], errors="coerce")

#drop fitur konstan jika ada 
const_cols = [c for c in feature_cols if df[c].nunique() == 1]
feature_cols = [c for c in feature_cols if c not in const_cols]

X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
y = df[target_col].astype(int)

# split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X_train, y_train)

#evaluasi
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

#save
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n{rep}\n\nConfusion Matrix:\n{cm}\n")

with open("tree_rules.txt", "w") as f:
    f.write(export_text(clf, feature_names=feature_cols))

joblib.dump(clf, "model.pkl")
print("Model saved to model.pkl")
print("Done. See metrics.txt & tree_rules.txt")
