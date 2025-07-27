import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Carregar dados do CSV
df = pd.read_csv("landmarks.csv")

# 2. Separar recursos (X) e rótulos (y)
X = df.drop("label", axis=1)
y = df["label"]

# 3. Codificar rótulos
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Treinar modelo com TODOS os dados
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y_encoded)

# 6. Salvar modelo, codificador e scaler
joblib.dump(clf, "activity_recognition_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Modelo treinado com todos os dados e salvo com sucesso.")