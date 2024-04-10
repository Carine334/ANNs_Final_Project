from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import base64
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

app = Flask(__name__, static_url_path='/static')

# Load pre-trained model
model = load_model('traffic_signs_model.h5')
#print(model.summary())


# Define the directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variables
IMG_SIZE = 48
threshold = 0.5

# Label Overview
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


def preprocess_images(img):
    # Resize image to (48, 48) pixels
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to be in the range [0, 1]
    img = img.astype("float32") / 255.0
    return img


def predict_result(image):
    pred = model.predict(image)
    predicted_label = np.argmax(pred[0])
    confidence = pred[0][predicted_label]
    if confidence < threshold:
        predicted_class_name = "Not known traffic sign"
        confidence = 1 - confidence
    else:
        predicted_class_name = classes[predicted_label]
    return predicted_class_name, confidence


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, cam_path, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))  # Resize heatmap to match image size
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Convert PIL image to array
    img = keras.utils.img_to_array(img)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img

    # Convert the superimposed image back to PIL format
    superimposed_img_pil = Image.fromarray(superimposed_img.astype(np.uint8))

    # Save the superimposed image
    superimposed_img_pil.save(cam_path)

    # Return the file path of the superimposed image
    return cam_path

@app.route("/")
def start():
    return render_template("start.html")


@app.route('/webcam.html')
def webcam():
    return render_template('webcam.html')



@app.route('/upload', methods=['POST'])
def upload():
    last_conv_layer_name = "conv2d_11"
    if 'file' not in request.files:
        return jsonify(error="No file selected")

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected")

    # Check if the uploaded file is a video
    if file.filename.endswith(('mp4', 'avi', 'mov')):
        # Read the video file
        video_stream = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        predictions = []
        while True:
            ret, frame = video_stream.read()
            if not ret:
                break
            # Preprocess the frame
            preprocessed_frame = preprocess_images(frame)
            preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
            # Predict the result
            prediction, confidence = predict_result(preprocessed_frame)
            # Generate heatmap
            heatmap = make_gradcam_heatmap(preprocessed_frame, model, last_conv_layer_name)
            # Encode heatmap image as base64 string
            _, heatmap_buffer = cv2.imencode('.jpg', heatmap)
            heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')
            predictions.append((prediction, confidence, heatmap_base64))
        # Release video stream
        video_stream.release()
        return jsonify(predictions=predictions)

    else:
        # Handle image file
        if file.filename.endswith(('png', 'jpg', 'jpeg')):
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            preprocessed_image = preprocess_images(image)                    # Preprocess the image
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
            prediction, confidence = predict_result(preprocessed_image)
            # Generate heatmap
            heatmap = make_gradcam_heatmap(preprocessed_image, model, last_conv_layer_name)
            # Superimpose heatmap on the original image and save it
            cam_path = save_and_display_gradcam(image, heatmap, cam_path="static/cam.jpg", alpha=0.4)
            # Convert original image to base64 string
            _, img_buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(img_buffer).decode('utf-8')
            return jsonify(name_class=prediction, confidence_level=float(confidence), cam_path=cam_path, uploaded_image=img_base64)

    return jsonify(error="Invalid file format")


@app.route('/webcam_feed')
def webcam_feed():
    return render_template('webcam.html')


@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    last_conv_layer_name = "conv2d_11"
    # Retrieve the image data from the request
    data = request.form['image_data']
    # Decode the base64 encoded image data
    image_data = base64.b64decode(data.split(',')[1])
    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode the image array
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Preprocess the frame
    preprocessed_frame = preprocess_images(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
    # Predict the result
    prediction, confidence = predict_result(preprocessed_frame)
    # Generate heatmap
    heatmap = make_gradcam_heatmap(preprocessed_frame, model, last_conv_layer_name)
    # Superimpose heatmap on the original image and save it
    cam_path = save_and_display_gradcam(frame, heatmap, cam_path="static/cam.jpg", alpha=0.4)
    # Convert original image to base64 string
    _, img_buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_buffer).decode('utf-8')
    # Return the prediction result
    return jsonify(name_class=prediction, confidence_level=float(confidence), cam_path=cam_path, uploaded_image=img_base64)



if __name__ == "__main__":

    app.run(debug=True)



    return jsonify(name_class=prediction, confidence_level = float(confidence))

if __name__ == "__main__":
    app.run(debug=True)
