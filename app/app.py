import os
import pickle as pkl
import numpy as np
import pandas as pd
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

class MyClassfier(DecisionTreeClassifier):
    def __init__(self):
        super().__init__()
        self.dic = dict()
        
    def encodeStr(self, x):
        x = x.copy()
        for i,col in enumerate(x.columns):
            if str(x[col].dtype)!="object":
                continue
            dic = {_:i for i,_ in enumerate(list(set(x[col])))}
            x[col] = np.array([dic[_] for _ in x[col]])
            self.dic[i] = dic
        return x
    
    def fit(self, x, y):
        x = self.encodeStr(x)
        return super().fit(x,y)
        
    def convert_(self,x):
        try:
            float(x)
            return float(x)
        except:
            return -1

    def predict(self, datas):
        datas = [[self.dic.get(i_,{x_:self.convert_(x_)}).get(x_, -1) for i_,x_ in enumerate(data)] for data in datas]
        return super().predict(datas)



app = Flask(__name__)

app.url_map.converters['clf'] = MyClassfier

@app.route('/', methods=['GET', 'POST'])
def route_index():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model = MyClassfier()
        with open(f"app/watch.pkl",'rb') as file:
            model  = pkl.loads(file.read())
        predictions = model.predict([[myage, mygender]])
        return render_template('index.html', href2='The suitable watch for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))
    

@app.route('/watch', methods=['GET', 'POST'])
def route_watch():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('watch.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model = load('app/watch-recommender.joblib')
        predictions = model.predict([[myage, mygender]])
        return render_template('watch.html', href2='The suitable watch for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))

@app.route('/phone', methods=['GET', 'POST'])
def route_phone():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('phone.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model = load('app/phone-recommender.joblib')
        predictions = model.predict([[myage, mygender]])
        return render_template('phone.html', href2='The suitable phone for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))

@app.route('/music', methods=['GET', 'POST'])
def route_music():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('music.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        academic_qualification = request.form['academic-qualification']
        model = load('app/music-recommender.joblib')
        predictions = model.predict([[myage, mygender, academic_qualification]])
        return render_template('music.html', href2='The suitable music for you (age:'+str(myage)+' ,gender:'+str(mygender)+',academic qualification:'+str(academic_qualification)+') is:'+ str(predictions[0]))

@app.route('/travel', methods=['GET', 'POST'])
def route_travel():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('travel.html', href2='')
    else:
        mysalary = request.form['salary']
        mygender = request.form['gender']
        marital = request.form['marital']
        model = load('app/travel-recommender.joblib')
        predictions = model.predict([[mysalary, mygender, marital]])
        return render_template('travel.html', href2='The suitable place for you (salary:'+str(mysalary)+' ,gender:'+str(mygender)+',marital:'+str(marital)+') to travel is:'+ str(predictions[0]))

@app.route('/vehicle', methods=['GET', 'POST'])
def route_vehicle():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('vehicle.html', href2='')
    else:
        myage = request.form['age']
        mysalary = request.form['salary']
        myclass = request.form['class']
        model = load('app/vehicle-recommender.joblib')
        predictions = model.predict([[myage, mysalary, myclass]])
        return render_template('vehicle.html', href2='The suitable vehicle for you (age:'+str(myage)+' ,salary:'+str(mysalary)+',class:'+str(myclass)+') is:'+ str(predictions[0]))