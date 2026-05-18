import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Charger le datamart final
df = pd.read_excel("data_final.xlsx")

# Cible du modèle
target = "PCT Voix"

# Colonnes à exclure
cols_to_drop = [
    "IRIS",
    "code_iris",
    "candidat",
    "nom_candidat",
    "parti",
    "nuance",
    target
]

# Garder uniquement les colonnes utiles
X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
y = df[target]

# Transformer les colonnes texte en variables numériques
X = pd.get_dummies(X)

# Supprimer les lignes avec valeurs manquantes
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=[target])
y = data[target]

# Séparer entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Modèle Gradient Boosting
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Entraînement
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("R² :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))

# Exemple de prédictions
resultats = pd.DataFrame({
    "Valeur réelle": y_test,
    "Valeur prédite": y_pred
})
print("\n" + "="*50)
print("      RÉSULTATS GRADIENT BOOSTING")
print("="*50)

print(f"{'Métrique':<20} | {'Valeur'}")
print("-"*50)

print(f"{'R²':<20} | {r2_score(y_test, y_pred):.4f}")
print(f"{'MAE':<20} | {mean_absolute_error(y_test, y_pred):.6f}")
print(f"{'MSE':<20} | {mean_squared_error(y_test, y_pred):.6f}")

print("\n" + "="*50)
print("      EXEMPLES DE PRÉDICTIONS")
print("="*50)

resultats = pd.DataFrame({
    "Valeur réelle": y_test.values,
    "Valeur prédite": y_pred
})

print(resultats.head(10).to_string(index=False))
