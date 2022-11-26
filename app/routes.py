# @Time: 2022/11/25 9:30
from app import app
from flask import render_template


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Lenny'}
    return render_template('index.html', title='Home', user=user)


@app.route('/predict')
def predict():
    return render_template('predict.html', title='Predict')


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
