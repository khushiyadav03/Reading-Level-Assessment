import os
from flask import Flask, render_template, request
from src.preprocessing import clean_text
from src.features import extract_features
from src.utils import load_model, score_to_grade

app = Flask(__name__)
model = load_model("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    grade = None
    warning = None

    if request.method == "POST":
        text = request.form["content"]

        if len(text.split()) < 40:
            warning = "Text is short. Accuracy improves with longer content."

        clean = clean_text(text)
        features = extract_features(clean)

        score = model.predict([list(features.values())])[0]
        grade = score_to_grade(score)

    return render_template(
        "index.html",
        score=round(score, 2) if score is not None else None,
        grade=grade,
        warning=warning
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
