import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("output", exist_ok=True)

df = pd.read_csv("dataset/winequality-red.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Lasso(alpha=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

joblib.dump(model, "output/model.pkl")
json.dump(
    {"experiment": "EXP-03", "model": "Lasso(alpha=0.01)", "mse": mse, "r2_score": r2},
    open("output/results.json", "w"), indent=4
)
