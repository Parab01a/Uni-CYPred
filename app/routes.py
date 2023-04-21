# @Time: 2022/11/25 9:30
import os
import io
import base64
import heapq
import deepchem as dc
import numpy as np
from flask import render_template, request, redirect, url_for, jsonify
from joblib import load
from rdkit import Chem
from rdkit.Chem import DataStructs, Draw, MACCSkeys
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from scipy.spatial import distance

from app import app

model_2b6 = load('2b6_maccs_svm.pkl')
model_2c8 = load('2c8_mol2vec_svm.pkl')
x_train_2b6 = load('2b6_maccs_x_train.pkl')
x_train_2c8 = load('2c8_mol2vec_x_train.pkl')
x_train_2b6_scaled_smo = load('2b6_maccs_x_train_scaled_smo.pkl')
x_train_2c8_scaled = load('2c8_mol2vec_x_train_scaled.pkl')

ALLOWED_EXTENSIONS = {'sdf', 'txt'}  # 暂时只有sdf


def rdkit_numpy_convert(fp):
    output = []
    arr = np.zeros([1, ])
    DataStructs.ConvertToNumpyArray(fp, arr)
    output.append(arr)
    return np.asarray(output)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def my_enumerate(iterable, start=0):
    return zip(range(start, len(iterable) + start), iterable)


def application_domain_2b6(x_scaled):
    DT_2b6 = 11.199
    i = 0
    dis_2b6 = []
    while i < x_train_2b6_scaled_smo.shape[0]:
        dis_2b6.append(distance.euclidean(x_scaled, x_train_2b6_scaled_smo[i]))
        i += 1
    knn_dis_2b6 = heapq.nsmallest(3, dis_2b6)
    d_max_2b6 = np.max(knn_dis_2b6)
    if d_max_2b6 > DT_2b6:
        AD_2b6 = 'OD'
    else:
        AD_2b6 = 'ID'
    return AD_2b6


def application_domain_2c8(x_scaled):
    DT_2c8 = 15.254
    i = 0
    dis_2c8 = []
    while i < x_train_2c8_scaled.shape[0]:
        dis_2c8.append(distance.euclidean(x_scaled, x_train_2c8_scaled[i]))
        i += 1
    knn_dis_2c8 = heapq.nsmallest(3, dis_2c8)
    d_max_2c8 = np.max(knn_dis_2c8)
    if d_max_2c8 > DT_2c8:
        AD_2c8 = 'OD'
    else:
        AD_2c8 = 'ID'
    return AD_2c8


def is_valid_smiles(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            return True
        else:
            return False
    except:
        return False


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', title='Predict')


# predict all together but in results, user can choose which one they want
@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'GET':
        return render_template('results.html', title='Results')
    if request.method == 'POST':
        s = request.form.get("smiles", type=str)
        file = request.files.get("file_input")
        draw_s = request.form.get("draw_smiles", type=str)
        # file = request.files["file_input"]  # FileStorage type
        # if not is_valid_smiles(s):
        #     s = ''
        # else:
        #     pass
        print(s)
        print(file)
        # print(file.filename)
        print(draw_s)
        # 使用s提交：s为空，file为None
        if s == '' and file is None and draw_s is None:
            return redirect(url_for('predict'))
        # 使用file提交：file.filename为空（file不为空file为空的FileStorage对象），s为None
        # if file.filename == '' and s is None and draw_s is None:
        #     return redirect(url_for('predict'))
        if draw_s == '' and s is None and file is None:
            return redirect(url_for('predict'))
        if s or file or draw_s:
            if s:
                # single molecule prediction
                # 2b6
                m = Chem.MolFromSmiles(s)
                fp_2b6 = MACCSkeys.GenMACCSKeys(m)
                x_2b6 = rdkit_numpy_convert(fp_2b6)
                scale_2b6 = StandardScaler().fit(x_train_2b6)
                x_2b6_scaled = scale_2b6.transform(x_2b6)
                # here prediction is a 2d-array, extract the prediction value of class 1
                prediction_2b6 = np.round(model_2b6.predict_proba(x_2b6_scaled)[:, 1], 3)  # columns 2
                # convert to str
                prediction_str_2b6 = "".join(str(i) for i in prediction_2b6)
                prediction_float_2b6 = float(prediction_str_2b6)

                # 2c8
                featurizer = dc.feat.Mol2VecFingerprint()
                x_2c8 = featurizer.featurize(s)  # type(x_2c8): numpy.ndarray; x_2c8.shape: (1, 300)
                scale_2c8 = StandardScaler().fit(x_train_2c8)
                x_2c8_scaled = scale_2c8.transform(x_2c8)
                prediction_2c8 = np.round(model_2c8.predict_proba(x_2c8_scaled)[:, 1], 3)
                prediction_str_2c8 = "".join(str(i) for i in prediction_2c8)
                prediction_float_2c8 = float(prediction_str_2c8)

                imgByteArr = io.BytesIO()  # 储存二进制文件
                img = Draw.MolToImage(m, size=(320, 180))
                img.save(imgByteArr, format='PNG', dpi=(600, 600))
                imgByteArr.seek(0)  # 从0位开始读取
                plot_url = base64.b64encode(imgByteArr.getvalue()).decode()  # 将二进制文件转换为base64编码的字符串

                AD_2b6 = application_domain_2b6(x_2b6_scaled[0])
                AD_2c8 = application_domain_2c8(x_2c8_scaled[0])

                return render_template('results.html', title='Results', s=s, prediction_float_2b6=prediction_float_2b6,
                                       prediction_float_2c8=prediction_float_2c8, plot_url=plot_url, AD_2b6=AD_2b6,
                                       AD_2c8=AD_2c8)

            if file:
                filename = (secure_filename(filename=file.filename))
                # print(filename)
                sdf_data = file.read().decode('utf-8')  # the input sdf is string type

                # SDMolSupplier has 2 ways to read data: 1) From file directly: Chem.SDMolSupplier('.sdf')
                # 2) From string: The SetData method is used to parse the string passed to it as SDF-formatted data
                # and store it in the SDMolSupplier object. During this process, the SDMolSupplier object automatically
                # parses the SDF data into multiple Mol objects and stores them in a list so that we can iterate over
                # all the molecules.
                suppl = Chem.SDMolSupplier()
                suppl.SetData(sdf_data)
                # print(suppl)

                sdf_index_strList = []
                sdf_smiles_strList = []
                sdf_prediction_floatList_2b6 = []
                sdf_prediction_floatList_2c8 = []
                sdf_img = []
                sdf_AD_strList_2b6 = []
                sdf_AD_strList_2c8 = []
                # iterate over(遍历) all the molecules
                for sdf_index, sdf_mol in enumerate(suppl):
                    if sdf_mol is not None:
                        sdf_index = str(sdf_index)
                        sdf_smiles = str(Chem.MolToSmiles(sdf_mol))
                        sdf_index_strList.append(sdf_index)
                        sdf_smiles_strList.append(sdf_smiles)
                        # 2b6
                        sdf_fp_2b6 = MACCSkeys.GenMACCSKeys(sdf_mol)
                        sdf_x_2b6 = rdkit_numpy_convert(sdf_fp_2b6)
                        scale_2b6 = StandardScaler().fit(x_train_2b6)
                        sdf_x_2b6_scaled = scale_2b6.transform(sdf_x_2b6)
                        sdf_prediction_2b6 = np.round(model_2b6.predict_proba(sdf_x_2b6_scaled)[:, 1], 3)
                        sdf_prediction_str_2b6 = "".join(str(i) for i in sdf_prediction_2b6)
                        sdf_prediction_float_2b6 = float(sdf_prediction_str_2b6)
                        sdf_prediction_floatList_2b6.append(sdf_prediction_float_2b6)
                        # output the class:
                        # if sdf_prediction_2b6 > 0.6:
                        #     sdf_class_2b6 = "inhibitor"
                        # else:
                        #     sdf_class_2b6 = "non-inhibitor"
                        # sdf_prediction_strList_2b6.append((sdf_prediction_str_2b6, sdf_class_2b6))
                        # <p>Class: {{ sdf_prediction_str_2b6[1] }}</p>

                        # 2c8
                        sdf_x_smiles_2c8 = Chem.MolToSmiles(sdf_mol)
                        featurizer = dc.feat.Mol2VecFingerprint()
                        sdf_x_2c8 = featurizer.featurize(sdf_x_smiles_2c8)
                        scale_2c8 = StandardScaler().fit(x_train_2c8)
                        sdf_x_2c8_scaled = scale_2c8.transform(sdf_x_2c8)
                        sdf_prediction_2c8 = np.round(model_2c8.predict_proba(sdf_x_2c8_scaled)[:, 1], 3)
                        sdf_prediction_str_2c8 = "".join(str(i) for i in sdf_prediction_2c8)
                        sdf_prediction_float_2c8 = float(sdf_prediction_str_2c8)
                        sdf_prediction_floatList_2c8.append(sdf_prediction_float_2c8)

                        imgByteArr = io.BytesIO()  # 储存二进制文件
                        img = Draw.MolToImage(sdf_mol, size=(320, 180))
                        img.save(imgByteArr, format='PNG', dpi=(600, 600))
                        imgByteArr.seek(0)  # 从0位开始读取
                        plot_url = base64.b64encode(imgByteArr.getvalue()).decode()
                        sdf_img.append(plot_url)

                        sdf_AD_strList_2b6.append(application_domain_2b6(sdf_x_2b6_scaled[0]))
                        sdf_AD_strList_2c8.append(application_domain_2c8(sdf_x_2c8_scaled[0]))

                data = zip(sdf_index_strList, sdf_smiles_strList, sdf_img, sdf_prediction_floatList_2b6,
                           sdf_prediction_floatList_2c8, sdf_AD_strList_2b6, sdf_AD_strList_2c8)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                return render_template('results.html', title='Results', data=data)
            if draw_s:
                # 2B6
                draw_m = Chem.MolFromSmiles(draw_s)
                draw_fp_2b6 = MACCSkeys.GenMACCSKeys(draw_m)
                draw_x_2b6 = rdkit_numpy_convert(draw_fp_2b6)
                scale_2b6 = StandardScaler().fit(x_train_2b6)
                draw_x_2b6_scaled = scale_2b6.transform(draw_x_2b6)
                # here prediction is a 2d-array, extract the prediction value of class 1
                draw_prediction_2b6 = np.round(model_2b6.predict_proba(draw_x_2b6_scaled)[:, 1], 3)  # columns 2
                # convert to str
                draw_prediction_str_2b6 = "".join(str(i) for i in draw_prediction_2b6)
                draw_prediction_float_2b6 = float(draw_prediction_str_2b6)

                # 2c8
                featurizer = dc.feat.Mol2VecFingerprint()
                draw_x_2c8 = featurizer.featurize(draw_s)  # type(x_2c8): numpy.ndarray; x_2c8.shape: (1, 300)
                scale_2c8 = StandardScaler().fit(x_train_2c8)
                draw_x_2c8_scaled = scale_2c8.transform(draw_x_2c8)
                draw_prediction_2c8 = np.round(model_2c8.predict_proba(draw_x_2c8_scaled)[:, 1], 3)
                draw_prediction_str_2c8 = "".join(str(i) for i in draw_prediction_2c8)
                draw_prediction_float_2c8 = float(draw_prediction_str_2c8)

                imgByteArr = io.BytesIO()  # 储存二进制文件
                img = Draw.MolToImage(draw_m, size=(320, 180))
                img.save(imgByteArr, format='PNG', dpi=(600, 600))
                imgByteArr.seek(0)  # 从0位开始读取
                draw_plot_url = base64.b64encode(imgByteArr.getvalue()).decode()  # 将二进制文件转换为base64编码的字符串

                draw_AD_2b6 = application_domain_2b6(draw_x_2b6_scaled[0])
                draw_AD_2c8 = application_domain_2c8(draw_x_2c8_scaled[0])

                return render_template('results.html', title='Results', draw_s=draw_s, draw_prediction_float_2b6=draw_prediction_float_2b6,
                                       draw_prediction_float_2c8=draw_prediction_float_2c8, draw_plot_url=draw_plot_url,
                                       draw_AD_2b6=draw_AD_2b6, draw_AD_2c8=draw_AD_2c8)


@app.route('/about')
def about():
    return render_template('about.html', title='About')


