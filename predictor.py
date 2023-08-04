from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np
import os

model_path = "modelv1.h5"
model = load_model(model_path)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400
    if file:
        # Preprocess your image
        img = keras_image.load_img(file, target_size=(150, 150))
        img_tensor = keras_image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0

        # Predict the class of the image
        prediction = model.predict(img_tensor)

        binary_prediction = (prediction > 0.5).astype(int)

        # return it in json in this format { pneumonia: boolean; confidence: prediction }
        return jsonify(
            {
                "pneumonia": bool(binary_prediction[0][0]),
                "confidence": str(prediction[0][0]),
            }
        )
    #     return str(binary_prediction)
