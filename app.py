# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
import requests
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Define model file path
MODEL_PATH = "model/train_model.h5"
GDRIVE_FILE_ID = "1n9wZ2lXghE9D-OljI4zc81v8Fal5wecr"  # Replace with your file ID
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# Function to download the model from Google Drive if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # Create folder if not exists
        
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully!")

# Download model if not available
download_model()

# Load model
model = load_model(MODEL_PATH)
print('✅ Model loaded successfully!')

# Function to predict disease
def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(224, 224))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D

    result = model.predict(test_image).round(3)  # Predict disease
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # Get the index of max value

    if pred == 0:
        return "Gray leaf spot", 'gray.html'  # If index 0
    elif pred == 1:
        return 'Healthy Corn Plant', 'healthy_plant.html'  # If index 1
    elif pred == 2:
        return 'Northern Leaf Blight', 'blight.html'  # If index 2
    else:
        return "Common Rust", 'disease_plant.html'  # If index 3

# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input image from client, predict class, and render respective .html page
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Get input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user_uploaded', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create folder if not exists
        file.save(file_path)

        print("@@ Predicting class...")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=5000)  # Use port 5000
