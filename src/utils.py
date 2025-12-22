import pandas as pd
import pickle
import os


def load_csv(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)



def save_model(model, path="model.pkl"):
    
    with open(path, "wb") as f:
        pickle.dump(model, f)



def load_model(path="model.pkl"):
    
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found. Train the model first.")
    with open(path, "rb") as f:
        return pickle.load(f)



def score_to_grade(score):
    if score < -1.3:
        return "Very Easy (Grade 1–2)"
    elif score < -1.0:
        return "Easy (Grade 3–4)"
    elif score < -0.5:
        return "Medium (Grade 5–6)"
    elif score < 0.2:
        return "Hard (Grade 7–9)"
    else:
        return "Very Hard (Grade 10+)"




def text_statistics(text):
    
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')

    return {
        "word_count": len(words),
        "sentence_count": max(1, sentences),
        "avg_word_length": sum(len(w) for w in words) / max(1, len(words))
    }
