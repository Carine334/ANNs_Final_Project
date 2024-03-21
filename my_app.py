from flask import Flask, render_template, request, Response
from templates import *
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app= Flask(__name__, static_url_path='/static')


# Load pre-trained model
#model = load_model("traffic_signs_model.keras") 


@app.route("/")
def start():
    return render_template("start.html")

@app.route('/webcam.html')
def webcam():
    return render_template('webcam.html')



import traceback

@app.errorhandler(500)
def internal_error(exception):
    return "<pre>"+traceback.format_exc()+"</pre>"
