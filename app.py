from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('weights.h5')
def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = preprocess_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if prediction > 0.5:
        return render_template('index.html',pred = "Magninant")
    else:
        return render_template('index.html',pred = "Benign")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
