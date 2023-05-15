# @Time: 2022/11/25 9:30
import os
import io
import base64
import heapq
import uuid
import json
import datetime
import csv
import deepchem as dc
import numpy as np
import pandas as pd
from flask import render_template, request, redirect, url_for, jsonify, session, abort, make_response
from flask_cors import *
from joblib import load
from rdkit import Chem
from rdkit.Chem import DataStructs, Draw, MACCSkeys
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
from scipy.spatial import distance

from app import app, mysql_util

model_2b6 = load('2b6_maccs_svm.pkl')
model_2c8 = load('2c8_mol2vec_svm.pkl')
x_train_2b6 = load('2b6_maccs_x_train.pkl')
x_train_2c8 = load('2c8_mol2vec_x_train.pkl')
x_train_2b6_scaled_smo = load('2b6_maccs_x_train_scaled_smo.pkl')
x_train_2c8_scaled = load('2c8_mol2vec_x_train_scaled.pkl')

ALLOWED_EXTENSIONS = {'sdf', 'txt'}  # 暂时只有sdf
CORS(app, supports_credentials=True)  # 跨域


# db = mysql_util.MysqlUtil()  # 实例化 MySQL


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
    DT_2b6 = 19.599
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
    DT_2c8 = 15.738
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

                # 2c8
                featurizer = dc.feat.Mol2VecFingerprint()
                x_2c8 = featurizer.featurize(s)  # type(x_2c8): numpy.ndarray; x_2c8.shape: (1, 300)
                scale_2c8 = StandardScaler().fit(x_train_2c8)
                x_2c8_scaled = scale_2c8.transform(x_2c8)
                prediction_2c8 = np.round(model_2c8.predict_proba(x_2c8_scaled)[:, 1], 3)
                prediction_str_2c8 = "".join(str(i) for i in prediction_2c8)

                imgByteArr = io.BytesIO()  # 储存二进制文件
                img = Draw.MolToImage(m, size=(320, 180))
                img.save(imgByteArr, format='PNG', dpi=(600, 600))
                imgByteArr.seek(0)  # 从0位开始读取
                plot_url = base64.b64encode(imgByteArr.getvalue()).decode()  # 将二进制文件转换为base64编码的字符串
                # Question: 为什么图片不够清晰

                AD_2b6 = application_domain_2b6(x_2b6_scaled[0])
                AD_2c8 = application_domain_2c8(x_2c8_scaled[0])

                uuid_str = str(uuid.uuid4())
                command = 's'

                db = mysql_util.MysqlUtil()
                sql_insert = "INSERT INTO s_prediction_results(uuid_str, s, prediction_str_2b6, prediction_str_2c8, " \
                             "plot_url, AD_2b6, AD_2c8, create_date) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', NOW())" \
                             % (uuid_str, s, prediction_str_2b6, prediction_str_2c8, plot_url, AD_2b6, AD_2c8)
                db.insert(sql_insert)

                return redirect(
                    url_for('show_result', uuid_str=uuid_str, _method='GET', _external=True, _command=command))

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

                # 2c8
                featurizer = dc.feat.Mol2VecFingerprint()
                draw_x_2c8 = featurizer.featurize(draw_s)  # type(x_2c8): numpy.ndarray; x_2c8.shape: (1, 300)
                scale_2c8 = StandardScaler().fit(x_train_2c8)
                draw_x_2c8_scaled = scale_2c8.transform(draw_x_2c8)
                draw_prediction_2c8 = np.round(model_2c8.predict_proba(draw_x_2c8_scaled)[:, 1], 3)
                draw_prediction_str_2c8 = "".join(str(i) for i in draw_prediction_2c8)

                imgByteArr = io.BytesIO()  # 储存二进制文件
                img = Draw.MolToImage(draw_m, size=(320, 180))
                img.save(imgByteArr, format='PNG', dpi=(600, 600))
                imgByteArr.seek(0)  # 从0位开始读取
                draw_plot_url = base64.b64encode(imgByteArr.getvalue()).decode()  # 将二进制文件转换为base64编码的字符串

                draw_AD_2b6 = application_domain_2b6(draw_x_2b6_scaled[0])
                draw_AD_2c8 = application_domain_2c8(draw_x_2c8_scaled[0])

                uuid_str = str(uuid.uuid4())
                command = 'd'

                db = mysql_util.MysqlUtil()
                sql_insert = "INSERT INTO draw_prediction_results(uuid_str, draw_s, draw_prediction_str_2b6, " \
                             "draw_prediction_str_2c8, draw_plot_url, draw_AD_2b6, draw_AD_2c8, create_date) VALUES " \
                             "('%s', '%s', '%s', '%s', '%s', '%s', '%s', NOW())" % \
                             (uuid_str, draw_s, draw_prediction_str_2b6, draw_prediction_str_2c8, draw_plot_url,
                              draw_AD_2b6, draw_AD_2c8)
                db.insert(sql_insert)

                return redirect(
                    url_for('show_result', uuid_str=uuid_str, _method='GET', _external=True, _command=command))

            if file:
                filename = (secure_filename(filename=file.filename))
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                sdf_data = file.read().decode('utf-8')  # the input sdf is string type

                # SDMolSupplier has 2 ways to read data: 1) From file directly: Chem.SDMolSupplier('.sdf')
                # 2) From string: The SetData method is used to parse the string passed to it as SDF-formatted data
                # and store it in the SDMolSupplier object. During this process, the SDMolSupplier object automatically
                # parses the SDF data into multiple Mol objects and stores them in a list so that we can iterate over
                # all the molecules.
                suppl = Chem.SDMolSupplier()
                suppl.SetData(sdf_data)
                print(suppl)

                sdf_index_strList = []
                sdf_smiles_strList = []
                sdf_img = []
                sdf_prediction_strList_2b6 = []
                sdf_prediction_strList_2c8 = []
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
                        sdf_prediction_strList_2b6.append(sdf_prediction_str_2b6)
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
                        sdf_prediction_strList_2c8.append(sdf_prediction_str_2c8)

                        imgByteArr = io.BytesIO()  # 储存二进制文件
                        img = Draw.MolToImage(sdf_mol, size=(320, 180))
                        img.save(imgByteArr, format='PNG', dpi=(600, 600))
                        imgByteArr.seek(0)  # 从0位开始读取
                        plot_url = base64.b64encode(imgByteArr.getvalue()).decode()
                        sdf_img.append(plot_url)

                        sdf_AD_strList_2b6.append(application_domain_2b6(sdf_x_2b6_scaled[0]))
                        sdf_AD_strList_2c8.append(application_domain_2c8(sdf_x_2c8_scaled[0]))

                # convert list to json
                sdf_index_strList_json = json.dumps(sdf_index_strList)
                sdf_smiles_strList_json = json.dumps(sdf_smiles_strList)
                sdf_img_json = json.dumps(sdf_img)
                sdf_prediction_strList_2b6_json = json.dumps(sdf_prediction_strList_2b6)
                sdf_prediction_strList_2c8_json = json.dumps(sdf_prediction_strList_2c8)
                sdf_AD_strList_2b6_json = json.dumps(sdf_AD_strList_2b6)
                sdf_AD_strList_2c8_json = json.dumps(sdf_AD_strList_2c8)

                uuid_str = str(uuid.uuid4())
                command = 'f'

                db = mysql_util.MysqlUtil()
                sql_mode = "SET sql_mode='NO_BACKSLASH_ESCAPES';"  # 禁用MySQL的反斜杠自动转义
                db.cursor.execute(sql_mode)
                sql_insert = "INSERT INTO sdf_prediction_results(uuid_str, sdf_index_strList, sdf_smiles_strList, " \
                             "sdf_img, sdf_prediction_strList_2b6, sdf_prediction_strList_2c8, sdf_AD_strList_2b6, " \
                             "sdf_AD_strList_2c8, create_date) VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', NOW())" \
                             % (uuid_str, sdf_index_strList_json, sdf_smiles_strList_json, sdf_img_json,
                                sdf_prediction_strList_2b6_json, sdf_prediction_strList_2c8_json,
                                sdf_AD_strList_2b6_json, sdf_AD_strList_2c8_json)
                db.insert(sql_insert)

                return redirect(
                    url_for('show_result', uuid_str=uuid_str, _method='GET', _external=True, _command=command))


@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'GET':
        return render_template('results.html', title='Results')


@app.route('/results/<uuid_str>', methods=["GET"])
def show_result(uuid_str):
    command = request.args.get('_command')
    if command == 's':
        db = mysql_util.MysqlUtil()
        sql_select = "SELECT * FROM s_prediction_results WHERE uuid_str = %s"
        row = db.select(sql_select, (uuid_str,))  # 确保 uuid_str 被当做一个元素，而不是字符串中的多个字符
        if not row:
            abort(404)  # 数据库中记录已被删除
        uuid_str = row[1]
        s = row[2]
        prediction_str_2b6 = row[3]
        prediction_str_2c8 = row[4]
        plot_url = row[5]
        AD_2b6 = row[6]
        AD_2c8 = row[7]

        return render_template('results.html', title='Results', uuid_str=uuid_str, s=s,
                               prediction_str_2b6=prediction_str_2b6, prediction_str_2c8=prediction_str_2c8,
                               plot_url=plot_url, AD_2b6=AD_2b6, AD_2c8=AD_2c8)
    if command == 'd':
        db = mysql_util.MysqlUtil()
        sql_select = "SELECT * FROM draw_prediction_results WHERE uuid_str = %s"
        row = db.select(sql_select, uuid_str)
        if not row:
            abort(404)
        uuid_str = row[1]
        draw_s = row[2]
        draw_prediction_str_2b6 = row[3]
        draw_prediction_str_2c8 = row[4]
        draw_plot_url = row[5]
        draw_AD_2b6 = row[6]
        draw_AD_2c8 = row[7]

        return render_template('results.html', title='Results', uuid_str=uuid_str, draw_s=draw_s,
                               draw_prediction_str_2b6=draw_prediction_str_2b6,
                               draw_prediction_str_2c8=draw_prediction_str_2c8, draw_plot_url=draw_plot_url,
                               draw_AD_2b6=draw_AD_2b6, draw_AD_2c8=draw_AD_2c8)
    if command == 'f':
        db = mysql_util.MysqlUtil()
        sql_select = "SELECT * FROM sdf_prediction_results WHERE uuid_str = %s"
        row = db.select(sql_select, uuid_str)
        if not row:
            abort(404)
        uuid_str = row[1]
        sdf_index_strList = json.loads(row[2])
        sdf_smiles_strList = json.loads(row[3])
        sdf_img = json.loads(row[4])
        sdf_prediction_strList_2b6 = json.loads(row[5])
        sdf_prediction_strList_2c8 = json.loads(row[6])
        sdf_AD_strList_2b6 = json.loads(row[7])
        sdf_AD_strList_2c8 = json.loads(row[8])
        data = zip(sdf_index_strList, sdf_smiles_strList, sdf_img, sdf_prediction_strList_2b6,
                   sdf_prediction_strList_2c8, sdf_AD_strList_2b6, sdf_AD_strList_2c8)
        return render_template('results.html', title='Results', uuid_str=uuid_str, data=data)


@app.route('/about')
def about():
    return render_template('about.html', title='About')


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html', title='404'), 404


# 将结果从mysql中读取然后生成内存的csv文件，再用flask的send_file
@app.route('/download_csv/<table_name>/<uuid_str>', methods=['GET'])
def download_csv(table_name, uuid_str):
    if table_name == 's_prediction_results' or table_name == 'draw_prediction_results':
        db = mysql_util.MysqlUtil()
        sql_select = "SELECT * FROM {} WHERE uuid_str = %s".format(table_name)
        row = db.select(sql_select, (uuid_str,))
        if not row:
            abort(404)  # 数据库中记录已被删除
        s = row[2]
        prediction_str_2b6 = row[3]
        prediction_str_2c8 = row[4]
        AD_2b6 = row[6]
        AD_2c8 = row[7]
        create_date = row[8].strftime("%Y-%m-%d %H:%M:%S")
        data = [(s, prediction_str_2b6, prediction_str_2c8, AD_2b6, AD_2c8, create_date), ]
        df = pd.DataFrame(data,
                          columns=['Input SMILES', 'Prediction CYP2B6', 'Prediction CYP2C8', 'Domain CYP2B6',
                                   'Domain CYP2C8', 'Create Date'])
        df.index.name = 'Index'

        # pd生成csv文件并储存在内存中
        csv_file = io.StringIO()
        df.to_csv(csv_file)
        csv_file.seek(0)

        # 创建响应对象
        response = make_response(csv_file)
        response.headers.set('Content-Disposition', 'attachment', filename='prediction.csv')
        response.headers.set('Content-Type', 'text/csv')

        return response

    elif table_name == 'sdf_prediction_results':
        db = mysql_util.MysqlUtil()
        sql_select = "SELECT * FROM {} WHERE uuid_str = %s".format(table_name)
        row = db.select(sql_select, (uuid_str,))
        if not row:
            abort(404)
        sdf_smiles_strList = json.loads(row[3])
        sdf_prediction_strList_2b6 = json.loads(row[5])
        sdf_prediction_strList_2c8 = json.loads(row[6])
        sdf_AD_strList_2b6 = json.loads(row[7])
        sdf_AD_strList_2c8 = json.loads(row[8])
        create_date = pd.DataFrame({'Create Date': pd.Series(row[9].strftime("%Y-%m-%d %H:%M:%S"))})

        df = pd.concat([pd.DataFrame(sdf_smiles_strList, columns=['Input SMILES']),
                        pd.DataFrame(sdf_prediction_strList_2b6, columns=['Prediction CYP2B6']),
                        pd.DataFrame(sdf_prediction_strList_2c8, columns=['Prediction CYP2C8']),
                        pd.DataFrame(sdf_AD_strList_2b6, columns=['Domain CYP2B6']),
                        pd.DataFrame(sdf_AD_strList_2c8, columns=['Domain CYP2C8'])], axis=1)
        df = pd.concat([df, create_date], axis=1)
        df.index.name = 'Index'

        csv_file = io.StringIO()
        df.to_csv(csv_file)
        csv_file.seek(0)

        response = make_response(csv_file)
        response.headers.set('Content-Disposition', 'attachment', filename='prediction.csv')
        response.headers.set('Content-Type', 'text/csv')

        return response
