from flask import Flask,request,render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def plot_decision_percentage(score):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    y=['Adidas','Balenciaga' ,'Nike', 'Puma']
    y_axis = [0,0.5,1,1.5]
    score = np.squeeze(score)
    # getting values against each value of y
    plt.barh(y_axis, score,0.5)
    plt.yticks(y_axis,y)
    # setting label of y-axis
    plt.ylabel("brand")
    # setting label of x-axis
    plt.xlabel("percentage") 
    for bar, percen in zip(ax.patches, score.astype(np.float32)*100):
        if percen/200>0.2:
            ax.text(percen/200, bar.get_y()+bar.get_height()/2, str(percen)+"%", color = 'white', ha = 'center', va = 'center') 
    plt.savefig("D:\Download\DoAnChuyenNganh\static\output\\bar.png")

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    class_name = ["Adidas","Balenciaga","Nike","Puma"]
    model = keras.models.load_model("D:\Download\DoAnChuyenNganh\model\Mobilenet.h5")
    img = request.files["data"]
    img = Image.open(img)
    img.save("D:\Download\DoAnChuyenNganh\static\output\origin_image.png")
    img = img.resize((224,224))
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predict = model.predict(img_array)
    score = predict
    print("score: ",score)
    plot_decision_percentage(score=score)
    brand = class_name[np.argmax(score)]
    return render_template("predict.html",brand = brand)