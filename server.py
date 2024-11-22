from flask import Flask, render_template, request, send_file
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Folders
IMAGE_FOLDER = 'static/images'  # Pre-uploaded images
RESULT_FOLDER = 'results'

# Ensure results folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
model = YOLO('model/best.pt')

@app.route('/')
def index():
    # List all pre-uploaded images
    images = os.listdir(IMAGE_FOLDER)
    images = [os.path.join(IMAGE_FOLDER, img) for img in images]
    return render_template('index.html', images=images)

@app.route('/process', methods=['POST'])
def process_image():
    # Get the image path from the form
    image_path = request.form.get('image_path')

    if not image_path:
        return "No image selected", 400

    # Run inference using YOLOv8
    results = model(image_path)
    results_img = results[0].plot()  # Render detection results
    result_img_path = os.path.join(RESULT_FOLDER, f'result_{os.path.basename(image_path)}')
    Image.fromarray(results_img).save(result_img_path)

    # Serve the processed image
    return send_file(result_img_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
