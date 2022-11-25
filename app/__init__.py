# @Time: 2022/11/25 9:28
from flask import Flask

app = Flask(__name__)


from app import routes
