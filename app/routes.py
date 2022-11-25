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
    pass


@app.route('/contact')
def contact():
    pass
