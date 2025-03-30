from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    X_input = np.array(data["X"]).reshape(-1, 1)
    prediction = model.predict(X_input).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
