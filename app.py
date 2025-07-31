from flask import Flask, render_template, request, send_file
import os
import pickle
import json
import uuid
from utils.extractor import extract_text_from_file

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    input_text = None
    json_file = None
    txt_file = None

    if request.method == "POST":
        if request.form.get("news_text"):
            input_text = request.form.get("news_text")
        elif "file" in request.files:
            file = request.files["file"]
            if file:
                input_text = extract_text_from_file(file)

        if input_text:
            transformed = vectorizer.transform([input_text])
            pred = model.predict(transformed)[0]
            prob = model.predict_proba(transformed).max()
            prediction = "Fake News" if pred == 1 else "Real News"
            confidence = round(prob * 100, 2)

            # Save output files
            uid = str(uuid.uuid4())
            json_file = f"{uid}.json"
            txt_file = f"{uid}.txt"
            with open(f"static/{json_file}", "w") as jf:
                json.dump({
                    "input": input_text[:100] + "...",
                    "prediction": prediction,
                    "confidence": confidence
                }, jf, indent=4)
            with open(f"static/{txt_file}", "w", encoding="utf-8") as tf:
                tf.write(f"Prediction: {prediction}\nConfidence: {confidence}%\n\nText:\n{input_text}")

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           text=input_text,
                           json_file=json_file,
                           txt_file=txt_file)

@app.route("/download/<filename>")
def download(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
