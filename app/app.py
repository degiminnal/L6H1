from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href2='')
    else:
        myage = int(request.form['age'])
        mygender = int(request.form['gender'])
        model = load('app/watch-recommender.joblib')
        predictions = model.predict([[myage, mygender]])
        return render_template('index.html', href2='The suitable bread for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))

