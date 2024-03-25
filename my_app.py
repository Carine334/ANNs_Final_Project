from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
import os

app = Flask(__name__, static_url_path='/static')

# Load pre-trained model
model = load_model("model_aug.h5")

# Define the directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variables
IMG_SIZE = 48
threshold= 0.995

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

def preprocess_images(img):
    # Resize image to (48, 48) pixels
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to be in the range [0, 1]
    img = img.astype("float32") / 255.0
    return img

# Function to generate Grad-CAM visualization
def generate_grad_cam(model, last_conv_layer_name, image_tensor):
    # Get the model without the last softmax layer
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    # Get the gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_index = tf.argmax(predictions, axis=1)
        loss = predictions[:, class_index]

    # Get the gradients of the last conv layer with respect to the loss
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the class
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# Modify the predict_result function to also return Grad-CAM visualization
def predict_result(image):
    pred = model.predict(image)
    predicted_label = np.argmax(pred[0], axis=-1)
    confidence = pred[0][predicted_label]
    
    if confidence < threshold:
        predicted_class_name = "Not known traffic"
        confidence = 1 - confidence
        grad_cam = None
    else:
        predicted_class_name = classes[predicted_label]
        
        # Generate Grad-CAM
        last_conv_layer_name = 'dense_3'  # Change this to the name of your last convolutional layer
        heatmap = generate_grad_cam(model, last_conv_layer_name, image)
        
        # Resize heatmap to match the image size
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        grad_cam = cv2.addWeighted(image[0], 0.5, heatmap, 0.5, 0)
    
    return predicted_class_name, confidence, grad_cam


@app.route("/")
def start():
    return render_template("start.html")

@app.route('/webcam.html')
def webcam():
    return render_template('webcam.html')


# Update the upload route to include Grad-CAM visualization in the response
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(error="No file selected")
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected")

    # Save the uploaded file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the saved image
    image = cv2.imread(file_path)

    # Preprocess the image
    preprocessed_image = preprocess_images(image)

    # Predict the result
    prediction, confidence, grad_cam = predict_result(np.expand_dims(preprocessed_image, axis=0))

    return jsonify(name_class=prediction, confidence_level=float(confidence), grad_cam=grad_cam)


if __name__ == "__main__":
    app.run(debug=True)
