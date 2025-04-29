from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask App
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Get the absolute path of the current script
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")  # Correct template folder path
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")  # Ensure correct upload path

app = Flask(__name__, template_folder=TEMPLATE_DIR)  # Use absolute path for templates

# Ensure Upload Folder Exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Trained Model
MODEL_PATH = os.path.join(BASE_DIR, "breast_cancer_model.h5")  # Correct model path
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Image Size
IMG_SIZE = 150  # Same size used during training

# Function to Predict
def predict_cancer(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Get Prediction
    prediction = model.predict(img)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"
    confidence = float(prediction) if prediction > 0.5 else (1 - float(prediction))
    confidence *= 100  # Convert to percentage

    return result, confidence

# Homepage Route
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded!"})

            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No selected file!"})

            # Save Uploaded File
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Get Prediction
            result, confidence = predict_cancer(filepath)

            return render_template("index.html", uploaded_image=f"/static/uploads/{filename}", result=result, confidence=confidence)

        return render_template("index.html")

    except Exception as e:
        print(f"❌ Error in index route: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
