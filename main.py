from flask import Flask,request,render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    model = keras.models.load_model("D:\Download\DoAnChuyenNganh\model\Mobilenet.h5")
    img = request.files["data"]
    img = Image.open(img)
    img = img.resize((224,224))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predict = model.predict(img_array)
    return render_template("predict.html")