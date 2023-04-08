# @Time: 2022/11/25 9:28
from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config.from_object(Config)
app.config['UPLOAD_FOLDER'] = './app/static/upload'  # the program run from PredCyp.py
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10M

from app import routes
