from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model("modelo_mnist.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    a = request.files["image"]
    a.save("image.jpg")
    img = image.load_img("image.jpg", target_size=(28,28), color_mode="grayscale")
    img = image.img_to_array(img)
    copy = img.copy()
    img[img > 128] = 0
    img[copy <= 128] = 255
    img = img/img.max()
    prediction = model.predict(img.reshape(len(img), 28, 28, 1))
    return render_template("index.html", result=np.argmax(prediction))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)