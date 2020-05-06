#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import flask
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from numpy import asmatrix,asarray
import PIL
from PIL import Image
import numpy as np


# In[ ]:


app = Flask(__name__, template_folder='templates')
def init():
    global model
    with open(f'model/model.pkl','rb') as f:
        model = pickle.load(f)


# In[ ]:


@app.route('/')
def upload_file():
    return render_template('main.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
    if request.method == 'POST':
        image = Image.open(request.files['file'].stream).convert("L")
        image = asarray(image)
        gr_im= Image.fromarray(image)
        load_img = np.matrix(gr_im.resize((28,28)))
        img = Image.fromarray(load_img)
        load_image = asmatrix(img)
        load_image = load_image.flatten()
        prediction = model.predict(load_image)
        pred = 'Predicted Number: ' + str(prediction[0])
        return render_template('main.html', prediction_no ='Predicted number: {}'.format(prediction))
    
if __name__ == '__main__':
    init()
    app.run()

