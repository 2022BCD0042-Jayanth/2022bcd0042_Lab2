import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("output", exist_ok=True)

df = pd.read_csv("dataset/winequality-red.csv", sep=";")

selected_features = [
    "alcohol", "sulphates", "citric acid", "volatile acidity"
]

X = df[selected_features]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

joblib.dump(model, "output/model.pkl")
json.dump(
    {"experiment": "EXP-07", "model": "RF(SelectedFeatures)", "mse": mse, "r2_score": r2},
    open("output/results.json", "w"), indent=4
)
