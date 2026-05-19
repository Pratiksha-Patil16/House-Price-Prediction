from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    features = np.array([[area, bedrooms, bathrooms, stories, parking]])

    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹ {output}"
    )


if __name__ == "__main__":
    app.run(debug=True)
