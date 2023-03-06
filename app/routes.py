# @Time: 2022/11/25 9:30
from app import app
from flask import render_template, request, redirect, url_for
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from joblib import load
import numpy as np

svm = load('2c8_svm_ecfp4.pkl')


def rdkit_numpy_convert(fp):
    output = []
    arr = np.zeros([1, ])
    DataStructs.ConvertToNumpyArray(fp, arr)
    output.append(arr)
    return np.asarray(output)


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Lenny'}
    return render_template('index.html', title='Home', user=user)


@app.route('/predict')
def predict():
    return render_template('predict.html', title='Predict')


@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'GET':
        return render_template('results.html', title='Results')
    if request.method == 'POST':
        s = request.form.get("smiles", type=str)
        m = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2)
        x = rdkit_numpy_convert(fp)
        prediction = svm.predict(x)
        return render_template('results.html', title='Results', prediction='likely{}'.format(prediction))


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
