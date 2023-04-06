# @Time: 2022/11/25 9:30
from app import app
from flask import render_template, request, redirect, url_for
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.preprocessing import StandardScaler
from joblib import load
from PIL import Image
import numpy as np
import io
import base64

model = load('2c8_ecfp4_rf.pkl')
x_train = load('ecfp4_x_train.pkl')


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


# predict all together but in results, user can choose which one they want

@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'GET':
        return render_template('results.html', title='Results')
    if request.method == 'POST':
        # get SMILES of input molecule
        s = request.form.get("smiles", type=str)
        print(s)
        m = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2)
        x = rdkit_numpy_convert(fp)
        scale = StandardScaler().fit(x_train)
        x_scaled = scale.transform(x)
        # here prediction is a 2d-array, extract the prediction value of class 1
        prediction = model.predict_proba(x_scaled)[:, 1]
        # convert to str
        prediction = "".join(str(i) for i in prediction)

        imgByteArr = io.BytesIO()
        img = Draw.MolToImage(m, size=(500, 300))
        img.save(imgByteArr, format='PNG')
        imgByteArr.seek(0)

        plot_url = base64.b64encode(imgByteArr.getvalue()).decode()
        # if prediction == [1]:
        #     prediction = "inhibitor"
        # else:
        #     prediction = "non-inhibitor"

        return render_template('results.html', title='Results', prediction=prediction, plot_url=plot_url)


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
