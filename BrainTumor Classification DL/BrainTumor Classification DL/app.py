from flask import Flask, render_template, request, flash, redirect
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import base64
import io

app = Flask(__name__, static_url_path='/static', template_folder='templates')

# Load the trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Set input size
INPUT_SIZE = 64

def convert_image_to_base64(image_array):
    img_pil = Image.fromarray(image_array)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_base64}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Process the uploaded image with the model
            result, confidence_level, image_array = process_image(file)
            image_array_url = convert_image_to_base64(image_array)
            # Render the result template with the prediction, confidence level, and image array
            return render_template('result.html', result=result, confidence_level=confidence_level, image_array_url=image_array_url)
    return render_template('upload.html')

def process_image(file):
    try:
        # Read and preprocess the image
        image = Image.open(file)
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        image_array = np.array(image)
        input_img = np.expand_dims(image_array, axis=0)

        # Make predictions using the model
        result_probabilities = model.predict(input_img)
        predicted_class = np.argmax(result_probabilities)
        confidence_level = np.max(result_probabilities)  # Get the confidence level

        return predicted_class, confidence_level, image_array

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True, port=5002)
