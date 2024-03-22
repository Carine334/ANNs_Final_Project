from flask import Flask, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import color, exposure, transform

app = Flask(__name__, static_url_path='/static')

# Load pre-trained model
model = load_model("model_aug.h5")

@app.route("/")
def start():
    return render_template("start.html")

@app.route('/webcam.html')
def webcam():
    return render_template('webcam.html')


#variables
IMG_SIZE = 48

class UnetInferrer:
    def __init__(self):
        self.model = load_model('model_aug.h5')

    def get_prediction(self, image):
        # Preprocess image
        image = self.preprocess_image(image)
        # Perform prediction with the model
        prediction = self.model.predict(image)
        return prediction

    def preprocess_image(self, img):
        # Preprocess image here
        # For example:
        # Convert to HSV
        hsv = color.rgb2hsv(img)
        # Equalize histogram
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)
        # Resize
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE), mode='constant')
        return np.expand_dims(img, axis=0) # Add batch dimension

# Create an instance of UnetInferrer
m = UnetInferrer()

# Example usage:
image_data = cv2.imread('photo_5442879035244927474_y.jpg')
prediction = m.get_prediction(image_data)
print(prediction)




if __name__ == "__main__":
    app.run(debug=True)
