from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
from utils import predict_disease, CLASS_NAMES
import torch
import torch.nn as nn
from torchvision import models
import threading
import time

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Create the upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the trained model
def load_model(model_path):
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = nn.Conv2d(512, 23, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 23
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

MODEL_PATH = "best_squeezenet_1.pth"
model = load_model(MODEL_PATH)

# Function to delete the image after 10 minutes
def delete_image_after_timeout(file_path, timeout=600):  # Default timeout is 600 seconds (10 minutes)
    time.sleep(timeout)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted image: {file_path}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", prediction="No image uploaded.")
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", prediction="No image selected.")
        if file:
            # Save the uploaded image
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Start a background thread to delete the image after 10 minutes
            threading.Thread(target=delete_image_after_timeout, args=(file_path, 600), daemon=True).start()

            # Load image and predict
            image = Image.open(file_path).convert("RGB")
            prediction = predict_disease(model, image)
            image_url = f"/static/uploads/{file.filename}"  # path used in HTML
    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
