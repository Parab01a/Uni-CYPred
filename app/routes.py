# @Time: 2022/11/25 9:30
import os

from app import app
from flask import render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
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

ALLOWED_EXTENSIONS = {'sdf', 'txt', 'smi'}


def rdkit_numpy_convert(fp):
    output = []
    arr = np.zeros([1, ])
    DataStructs.ConvertToNumpyArray(fp, arr)
    output.append(arr)
    return np.asarray(output)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/predict')
def predict():
    return render_template('predict.html', title='Predict')


# predict all together but in results, user can choose which one they want

@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'GET':
        return render_template('results.html', title='Results')
    if request.method == 'POST':
        # single molecule prediction
        # get SMILES of input molecule
        s = request.form.get("smiles", type=str)
        print(s)
        m = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2)
        x = rdkit_numpy_convert(fp)
        scale = StandardScaler().fit(x_train)
        x_scaled = scale.transform(x)
        # here prediction is a 2d-array, extract the prediction value of class 1
        prediction = model.predict_proba(x_scaled)[:, 1]  # columns 2
        # convert to str
        prediction_str = "".join(str(i) for i in prediction)
        if prediction > 0.6:
            class_2c8 = "inhibitor"
        else:
            class_2c8 = "non-inhibitor"

        imgByteArr = io.BytesIO()
        img = Draw.MolToImage(m, size=(500, 300))
        img.save(imgByteArr, format='PNG')
        imgByteArr.seek(0)

        plot_url = base64.b64encode(imgByteArr.getvalue()).decode()

        if request.method == 'POST':
            # upload file function(batch prediction)
            # check if the post request has the file part
            if 'file_input' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files["file_input"]
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = (secure_filename(filename=file.filename))
                print(filename)
                print(type(file))
                content = file.read().decode('utf-8')
                print(type(content))
                # print(content)
                # mols = Chem.SDMolSupplier()
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template('results.html', title='Results', prediction_str=prediction_str, plot_url=plot_url,
                               class_2c8=class_2c8)


# @app.route('/results', methods=["GET", "POST"])
# def upload():
#         return render_template('results.html', title='Results')


@app.route('/about')
def about():
    return render_template('about.html', title='About')
