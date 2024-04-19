import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = load_model(r'C:\Users\hp\OneDrive\Desktop\BrainTumor Classification DL\BrainTumor10EpochsCategorical.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['imageFile']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                img = Image.fromarray(image)
                img = img.resize((64, 64))
                img = np.array(img)
                input_img = np.expand_dims(img, axis=0)
                result_probabilities = model.predict(input_img)
                predicted_class = np.argmax(result_probabilities)
                return render_template('index.html', predicted_class=predicted_class)
    return render_template('index.html', predicted_class=None)

if __name__ == '__main__':
    app.run(debug=True)