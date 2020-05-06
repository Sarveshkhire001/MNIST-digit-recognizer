import flask
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from numpy import asmatrix,asarray
import PIL
from PIL import Image
import numpy as np

with open(f'model/model.pkl','rb') as f:
        model = pickle.load(f)
    
app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def upload_image_file():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if request.method == 'POST':
        image = Image.open(request.files['file'].stream).convert("L")
        image = asarray(image)
        gr_im= Image.fromarray(image)
        load_img = np.matrix(gr_im.resize((28,28)))
        img = Image.fromarray(load_img)
        load_image = asmatrix(img)
        load_image = load_image.flatten()
        prediction = model.predict(load_image)
        pred = str(prediction[0])
        return render_template('main.html', prediction_no ='Predicted number: {}'.format(pred))
    
if __name__ == '__main__':
    app.run()