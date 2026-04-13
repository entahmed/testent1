import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ===============================
# 1️⃣ Charger Dataset
# ===============================
df = pd.read_csv(r"C:\Users\ahmed\OneDrive\Bureau\pfa_nthif\Approche 3\Dataset_extract.csv")
df.columns = df.columns.str.strip()
print("Shape initial:", df.shape)
print("Colonnes:", df.columns)

# ===============================
# 2️⃣ Nettoyage robuste
# ===============================
df["Group"] = df["Group"].astype(str).str.strip()
df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()
df["Sex"] = df["Sex"].replace({"male":1, "m":1, "female":0, "f":0})
print("Sex uniques après nettoyage:", df["Sex"].unique())

# ===============================
# 3️⃣ Sélection des Features
# ===============================
features = ["Hippo_Ratio_ICV", "Hippo_Asym_Index", "Age", "Sex"]
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes: {missing_cols}")

df_clean = df[features + ["Group"]].dropna()
print("Shape après nettoyage:", df_clean.shape)

if df_clean.shape[0] == 0:
    raise ValueError("Dataset vide après nettoyage → vérifier les données")

X = df_clean[features]
encoder = LabelEncoder()
y = encoder.fit_transform(df_clean["Group"])
class_names = encoder.classes_

# ===============================
# 4️⃣ Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ===============================
# 5️⃣ Preprocessing
# ===============================
num_features = ["Hippo_Ratio_ICV", "Hippo_Asym_Index", "Age"]
cat_features = ["Sex"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", "passthrough", cat_features)
])

# ===============================
# 6️⃣ Définition des Modèles
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
    "SVM": SVC(probability=True, class_weight="balanced", kernel="rbf")
}

# ===============================
# 7️⃣ Entraînement, évaluation & visualisation
# ===============================
for name, clf in models.items():
    model = Pipeline([("prep", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # ROC AUC multi-classe (OvR)
    auc = roc_auc_score(y_test, probs, multi_class="ovr")
    print("\n==============================")
    print(name)
    print("==============================")
    print(classification_report(y_test, preds, target_names=class_names))
    print("ROC AUC (OvR):", round(auc, 3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()

    # ROC Curve
    y_bin = label_binarize(y_test, classes=np.unique(y))
    plt.figure(figsize=(6,5))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.title(f"ROC Curve — {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# ===============================
# 8️⃣ Importance des Features (Random Forest)
# ===============================
rf_model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
])
rf_model.fit(X_train, y_train)

rf = rf_model.named_steps["clf"]
importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp)

plt.figure(figsize=(6,4))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance — Random Forest")
plt.show()