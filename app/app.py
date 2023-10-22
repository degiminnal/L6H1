import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

def convert_(x):
    try:
        float(x)
        return float(x)
    except:
        return -1

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def route_index():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([myage, mygender])]])
        return render_template('index.html', href2='The suitable watch for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))
    

@app.route('/watch', methods=['GET', 'POST'])
def route_watch():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('watch.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([myage, mygender])]])
        return render_template('watch.html', href2='The suitable watch for you (age:'+str(myage)+' ,gender:'+str(mygender)+') is:'+ str(predictions[0]))

@app.route('/phone', methods=['GET', 'POST'])
def route_phone():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('phone.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([myage, mygender])]])
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
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([myage, mygender, academic_qualification])]])
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
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([mysalary, mygender, marital])]])
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
        model, d = load(f"./app/watch-recommender.joblib")
        dic = {d[0][i]:{a:b for a,b in zip(d[1][i][0],d[1][i][1])} for i in range(len(d[0]))}
        predictions = model.predict([[dic.get(i_,{x_:convert_(x_)}).get(x_, -1) for i_,x_ in enumerate([myage, mysalary, myclass])]])
        return render_template('vehicle.html', href2='The suitable vehicle for you (age:'+str(myage)+' ,salary:'+str(mysalary)+',class:'+str(myclass)+') is:'+ str(predictions[0]))