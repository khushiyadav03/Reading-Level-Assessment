import os
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import clean_text
from src.features import extract_features
from src.utils import load_model


SAVE_DIR = r"C:\Users\punee\OneDrive\Desktop\Reading-Level-Assessment\results\graphs"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Saving graphs to:", SAVE_DIR)


df = pd.read_csv(r"C:\Users\punee\OneDrive\Desktop\Reading-Level-Assessment\data\raw\commonlitreadabilityprize.csv")
df = df.rename(columns={"excerpt": "text", "target": "score"})


df["clean_text"] = df["text"].apply(clean_text)

X = pd.DataFrame(df["clean_text"].apply(extract_features).tolist())
y = df["score"]


model = load_model("model.pkl")
preds = model.predict(X)


plt.figure()
plt.scatter(y, preds)
plt.xlabel("Actual Readability Score")
plt.ylabel("Predicted Readability Score")
plt.title("Actual vs Predicted Readability Scores")
plt.savefig(os.path.join(SAVE_DIR, "actual_vs_predicted.png"), dpi=300)
plt.close()


errors = y - preds

plt.figure()
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.savefig(os.path.join(SAVE_DIR, "error_distribution.png"), dpi=300)
plt.close()


importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_importance.png"), dpi=300)
plt.close()

print(" All graphs saved successfully!")
