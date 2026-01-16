import pandas as pd
from src.preprocessing import clean_text
from src.features import extract_features
from src.model import train_model
from src.utils import save_model


df = pd.read_csv("commonlitreadabilityprize/commonlitreadabilityprize.csv")

# Sample data for faster training
df = df.sample(n=min(500, len(df)), random_state=42)

df = df.rename(columns={"excerpt": "text", "target": "score"})


df["clean_text"] = df["text"].apply(clean_text)


feature_list = df["clean_text"].apply(extract_features)
X = pd.DataFrame(feature_list.tolist())
y = df["score"]


model, rmse = train_model(X, y)
print("Model trained successfully")
print("RMSE:", rmse)


save_model(model)
print("model.pkl saved successfully")
